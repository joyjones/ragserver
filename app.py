# video_service.py
from flask import Flask, request, jsonify
import subprocess
import os
import time
import hashlib
import shutil
from datetime import datetime
from pymongo import MongoClient
import requests
import threading
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from langchain_openai import ChatOpenAI
import json
import time as _time
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = Flask(__name__)

# ====== 可配置项 ======
MONGO_URI = 'mongodb://admin:admin123@localhost:27017/ragdata?authSource=admin'
DB_NAME = 'ragdata'
RAGDATA_DIR = 'E:\\Docker\\RAGData'  # 中间文件存放目录
SCAN_INTERVAL = 60  # 扫描间隔（秒）
N8N_MARKDOWN_URL = 'http://localhost:5678/webhook/convert/markdown'
N8N_VECTOR_URL = 'http://localhost:5678/webhook/convert/vector'
WHISPER_MODEL = 'large-v3'  # whisper模型名，可根据需要修改
MAPPING_FILE = os.path.join(RAGDATA_DIR, 'mapping.txt')
MESSAGE_LOG_FILE = os.path.join(RAGDATA_DIR, 'message_log.txt')

# ====== 向量化相关配置 ======
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "ragdata"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # 需在环境变量中设置

# ====== 向量数据库和LLM全局对象 ======
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
client = QdrantClient(url=QDRANT_URL, timeout=10, prefer_grpc=False)
vectorstore = Qdrant(
    client=client,
    collection_name=QDRANT_COLLECTION,
    embeddings=embeddings
)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

# ====== MongoDB连接 ======
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
dir_index_col = db['dir_index']
dir_file_state_col = db['dir_file_state']

# ====== proc_state枚举定义 ======
class ProcState:
    NEW = 0  # 新增文件
    AUDIO = 1  # 已处理为音频
    IMAGE = 2  # 已处理为图片
    RAW_TEXT = 3  # 已处理为原始文本
    MARKDOWN = 4  # 已处理为markdown可读文本
    VECTOR = 5  # 已生成向量数据

# ====== 工具函数 ======
def calc_md5(filepath, block_size=65536):
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(block_size), b''):
            md5.update(chunk)
    return md5.hexdigest()

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_file_state(file_path):
    stat = os.stat(file_path)
    return {
        'file_size': stat.st_size,
        'last_modify_time': int(stat.st_mtime),
    }

def list_files(dir_path, recursive, file_types):
    files = []
    if recursive:
        for root, _, filenames in os.walk(dir_path):
            for fname in filenames:
                if not file_types or os.path.splitext(fname)[1] in file_types:
                    files.append(os.path.join(root, fname))
    else:
        for fname in os.listdir(dir_path):
            fpath = os.path.join(dir_path, fname)
            if os.path.isfile(fpath) and (not file_types or os.path.splitext(fname)[1] in file_types):
                files.append(fpath)
    return files

def update_file_state(dir_id, file_name, state):
    dir_file_state_col.update_one(
        {'dir_id': dir_id, 'file_name': file_name},
        {'$set': state},
        upsert=True
    )

def get_file_unique_id(fpath):
    # 用md5做唯一id，避免中文名问题
    return calc_md5(fpath)

def process_video_to_audio(video_path, audio_path):
    cmd = ['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'mp3', audio_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def process_audio_to_text(audio_path, text_path):
    cmd = ['whisper', audio_path, '--model', WHISPER_MODEL, '--language', 'zh', '--output_format', 'txt', '--output_dir', os.path.dirname(text_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # whisper会生成同名txt，重命名到目标text_path
    base = os.path.splitext(os.path.basename(audio_path))[0]
    whisper_txt = os.path.join(os.path.dirname(text_path), base + '.txt')
    if os.path.exists(whisper_txt):
        shutil.move(whisper_txt, text_path)

def process_text_to_markdown(text_path, md_path):
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    resp = requests.post(N8N_MARKDOWN_URL, json={'text': text})
    resp.raise_for_status()
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(resp.json().get('markdown', ''))

def append_mapping(file_uid, file_path):
    # 追加一行到mapping.txt，格式：file_uid\tfile_path
    with open(MAPPING_FILE, 'a', encoding='utf-8') as f:
        f.write(f'{file_uid}\t{file_path}\n')

def request_markdown_async(file_uid, text_path):
    # 向n8n异步请求markdown生成
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    try:
        requests.post(N8N_MARKDOWN_URL, json={'file_uid': file_uid, 'text': text}, timeout=3)
    except Exception as e:
        print(f'n8n异步请求异常: {e}')

def process_md_to_vector(md_path, file_uid):
    # 1. 初始化embedding和Qdrant client
    client = QdrantClient(url=QDRANT_URL, timeout=10, prefer_grpc=False)
    # 2. 检查collection是否存在，不存在则创建
    if QDRANT_COLLECTION not in [c.name for c in client.get_collections().collections]:
        print(f"Qdrant中未找到collection '{QDRANT_COLLECTION}'，自动创建...")
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=rest.VectorParams(size=1536, distance=rest.Distance.COSINE)
        )
    # 3. 读取Markdown内容，分块后构建LangChain文档对象
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 使用RecursiveCharacterTextSplitter分块
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([content], metadatas=[{"file_uid": file_uid, "path": md_path}])
    # 检查是否已存在该file_uid，存在则删除（全量覆盖）
    search_result = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.file_uid",
                    match=rest.MatchValue(value=file_uid)
                )
            ]
        ),
        limit=1000
    )
    if search_result and search_result[0]:
        print(f"Qdrant已存在file_uid={file_uid}，执行upsert...")
        ids = [point.id for point in search_result[0]]
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=rest.PointIdsList(points=ids)
        )
    vectorstore.add_documents(docs)
    print(f"已分块写入Qdrant向量库: {file_uid}，共{len(docs)}块")

def main_loop():
    print('启动本地资料监控处理服务...')
    ensure_dir(RAGDATA_DIR)
    while True:
        now = int(time.time())
        print(f'开始扫描处理... 当前时间: {datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")}')
        for dir_cfg in dir_index_col.find({'enabled': True}):
            dir_path = dir_cfg['dirPath']
            recursive = dir_cfg.get('recursive', False)
            file_types = dir_cfg.get('fileTypes', [])
            dir_id = str(dir_cfg['_id'])
            notes = dir_cfg.get('notes', '')
            if not os.path.exists(dir_path):
                print(f'目录不存在: {dir_path}')
                continue
            files = list_files(dir_path, recursive, file_types)
            for fpath in files:
                fname = os.path.basename(fpath)
                fstate = get_file_state(fpath)
                file_uid = get_file_unique_id(fpath)
                # 检查是否已记录
                rec = dir_file_state_col.find_one({'dir_id': dir_id, 'file_name': fname})
                if not rec:
                    # 新文件
                    state = {
                        'dir_id': dir_id,
                        'file_name': fname,
                        'file_uid': file_uid,
                        'file_size': fstate['file_size'],
                        'file_type': os.path.splitext(fname)[1],
                        'file_md5': file_uid,
                        'proc_state': ProcState.NEW,
                        'last_proc_time': 0,
                        'last_modify_time': fstate['last_modify_time'],
                    }
                    update_file_state(dir_id, fname, state)
                    print(f'发现新文件: {fname}')
                    append_mapping(file_uid, fpath)
                else:
                    # 检查是否被修改
                    if rec.get('last_modify_time') != fstate['last_modify_time'] or rec.get('file_size') != fstate['file_size']:
                        print(f'文件被修改: {fname}')
                        update_file_state(dir_id, fname, {'proc_state': ProcState.NEW, 'last_modify_time': fstate['last_modify_time'], 'file_size': fstate['file_size']})
                # 推进处理流程（循环推进到不能再推进为止）
                while True:
                    rec = dir_file_state_col.find_one({'dir_id': dir_id, 'file_name': fname})
                    proc_state = rec.get('proc_state')
                    base_name = rec.get('file_uid', file_uid)
                    ext = os.path.splitext(fname)[1].lower()
                    rag_dir = os.path.join(RAGDATA_DIR, dir_id)
                    ensure_dir(rag_dir)
                    video_path = fpath
                    audio_path = os.path.join(rag_dir, base_name + '.mp3')
                    text_path = os.path.join(rag_dir, base_name + '.txt')
                    md_path = os.path.join(rag_dir, base_name + '.md')

                    # 检查中间文件实际存在性，必要时回退proc_state
                    real_state = ProcState.NEW
                    if os.path.exists(audio_path):
                        real_state = ProcState.AUDIO
                    if os.path.exists(text_path):
                        real_state = ProcState.RAW_TEXT
                    if os.path.exists(md_path):
                        real_state = ProcState.MARKDOWN
                    if proc_state == ProcState.VECTOR:
                        real_state = ProcState.VECTOR  # 保持VECTOR，不再重复写入
                    if proc_state != real_state:
                        update_file_state(dir_id, fname, {'proc_state': real_state})
                        proc_state = real_state

                    try:
                        if proc_state == ProcState.NEW and ext in ['.mp4', '.avi', '.mov']:
                            print(f'即将处理为音频: {fname} -> {audio_path}')
                            process_video_to_audio(video_path, audio_path)
                            update_file_state(dir_id, fname, {'proc_state': ProcState.AUDIO, 'last_proc_time': now})
                            print(f'已处理为音频: {fname}')
                        elif proc_state == ProcState.AUDIO:
                            print(f'即将处理为原始文本: {audio_path} -> {text_path}')
                            process_audio_to_text(audio_path, text_path)
                            update_file_state(dir_id, fname, {'proc_state': ProcState.RAW_TEXT, 'last_proc_time': now})
                            print(f'已处理为原始文本: {fname}')
                        elif proc_state == ProcState.RAW_TEXT:
                            print(f'即将异步请求n8n生成markdown: {text_path}')
                            request_markdown_async(base_name, text_path)
                            print(f'已请求n8n生成markdown，等待回调: {fname}')
                            break
                        elif proc_state == ProcState.MARKDOWN:
                            print(f'即将写入向量库: {md_path}')
                            process_md_to_vector(md_path, base_name)
                            update_file_state(dir_id, fname, {'proc_state': ProcState.VECTOR, 'last_proc_time': now})
                            print(f'已写入向量库: {fname}')
                        elif proc_state == ProcState.VECTOR:
                            break  # 已是最终状态，不再处理
                        else:
                            break  # 没有可推进的步骤，跳出循环
                    except Exception as e:
                        print(f'处理文件出错: {fname}, 错误: {e}')
                        break
        print(f'本轮扫描处理已完成，等待下次轮询...\n')
        time.sleep(SCAN_INTERVAL)

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'ok'})

@app.route('/save_markdown', methods=['POST'])
def save_markdown():
    data = request.json
    file_uid = data.get('file_uid')
    text = data.get('text')
    if not file_uid or not text:
        return jsonify({'status': 'fail', 'msg': '缺少file_uid或text'}), 400
    # 去除开头```markdown和结尾```
    lines = text.splitlines()
    if lines and lines[0].strip() == '```markdown':
        lines = lines[1:]
    if lines and lines[-1].strip() == '```':
        lines = lines[:-1]
    text = '\n'.join(lines)
    # 查找所有dir_id下的md_path
    for dir_cfg in dir_index_col.find({'enabled': True}):
        dir_id = str(dir_cfg['_id'])
        rag_dir = os.path.join(RAGDATA_DIR, dir_id)
        md_path = os.path.join(rag_dir, file_uid + '.md')
        if os.path.exists(rag_dir):
            # 保存markdown
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(text)
            # 更新状态
            rec = dir_file_state_col.find_one({'dir_id': dir_id, 'file_uid': file_uid})
            if rec:
                update_file_state(dir_id, rec['file_name'], {'proc_state': ProcState.MARKDOWN, 'last_proc_time': int(time.time())})
            print(f'收到n8n回调并保存markdown: {md_path}')
            return jsonify({'status': 'ok'})
    return jsonify({'status': 'fail', 'msg': '未找到file_uid'}), 404

def answer_question(question, top_k=3):
    docs = vectorstore.similarity_search(question, k=top_k)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"已知资料：\n{context}\n\n请根据上述资料回答：{question}"
    answer = llm.invoke(prompt)
    return answer.content, context

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    top_k = data.get('top_k', 3)
    answer, context = answer_question(question, top_k)
    return jsonify({"answer": answer, "context": context})

# ====== 飞书机器人配置 ======
FEISHU_APP_ID = os.getenv('FEISHU_APP_ID', 'cli_a8b6896dc7bad00e')
FEISHU_APP_SECRET = os.getenv('FEISHU_APP_SECRET', 'iWvgAla6BUnIQSPMbV6lPcolqR5frEpx')
_feishu_token_cache = {'token': None, 'expire': 0}

def get_tenant_access_token():
    now = _time.time()
    if _feishu_token_cache['token'] and _feishu_token_cache['expire'] > now:
        return _feishu_token_cache['token']
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal/"
    resp = requests.post(url, json={"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET})
    token = resp.json().get("tenant_access_token")
    expire = now + 7100  # 官方2小时，提前100秒刷新
    _feishu_token_cache['token'] = token
    _feishu_token_cache['expire'] = expire
    return token

def send_feishu_message(receive_id, content, msg_type="text", receive_id_type="chat_id"):
    token = get_tenant_access_token()
    url = f"https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type={receive_id_type}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8"
    }
    data = {
        "receive_id": receive_id,
        "msg_type": msg_type,
        "content": json.dumps({"text": content})
    }
    resp = requests.post(url, headers=headers, json=data, timeout=10)
    print(f'飞书API发送消息结果: {resp.text}')
    return resp.json()

def is_duplicate_message(message_id):
    if not os.path.exists(MESSAGE_LOG_FILE):
        return False
    with open(MESSAGE_LOG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == message_id:
                return True
    return False

def log_message_id(message_id):
    with open(MESSAGE_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(message_id + '\n')

@app.route('/feishu/event', methods=['POST'])
def feishu_event():
    data = request.json
    # 1. 服务器校验（飞书官方要求）
    type = data.get('type')
    if type == 'url_verification':
        return jsonify({'challenge': data.get('challenge')})
    # 2. 处理消息事件
    if type == 'event_callback':
        event = data.get('event', {})
        # 只处理文本消息
        if event.get('type') == 'message' and event.get('message_type') == 'text':
            message_id = event.get('message_id', '')
            if not message_id:
                return jsonify({'code': 0})
            if is_duplicate_message(message_id):
                print(f'检测到重复message_id: {message_id}，忽略处理')
                return jsonify({'code': 0})
            log_message_id(message_id)
            user_id = event.get('sender', {}).get('sender_id', {}).get('user_id')
            chat_id = event.get('chat_id')
            text = event.get('text', '')
            # 去除@机器人前缀
            if 'content' in event:
                import json as _json
                try:
                    content_obj = _json.loads(event['content'])
                    text = content_obj.get('text', text)
                except Exception:
                    pass
            # 调用本地问答
            try:
                answer, _ = answer_question(text, 3)
            except Exception as e:
                answer = f'处理出错: {e}'
            # 主动回复消息到飞书
            send_feishu_message(chat_id, answer, msg_type="text", receive_id_type="chat_id")
            print(f'飞书消息应答: chat_id={chat_id}, user_id={user_id}, Q={text}, A={answer}')
        return jsonify({'code': 0})
    else:
        # 其他类型事件也可支持问答
        event = data.get('event', {})
        message = event.get('message', {})
        chat_id = message.get('chat_id')
        message_id = message.get('message_id', '')
        if not message_id:
            return jsonify({'code': 0})
        if is_duplicate_message(message_id):
            print(f'检测到重复message_id: {message_id}，忽略处理')
            return jsonify({'code': 0})
        log_message_id(message_id)
        content_str = message.get('content', '')
        import json as _json
        try:
            content_obj = _json.loads(content_str)
            question = content_obj.get('text', '')
        except Exception:
            question = ''
        try:
            answer, _ = answer_question(question, 3)
        except Exception as e:
            answer = f'处理出错: {e}'
        if chat_id:
            send_feishu_message(chat_id, answer, msg_type="text", receive_id_type="chat_id")
        print(f'飞书事件通用问答: Q={question}, A={answer}')
        return jsonify({'code': 0})

if __name__ == '__main__':
    t = threading.Thread(target=main_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=9927)
