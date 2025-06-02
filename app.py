# video_service.py
from flask import Flask, request, jsonify
import subprocess
import os
import time
import hashlib
import shutil
import requests
import threading
import json
import time as _time
import pdfplumber
from datetime import datetime
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path

app = Flask(__name__)

# ====== 可配置项 ======
MONGO_URI = 'mongodb://admin:admin123@localhost:27017/ragdata?authSource=admin'
DB_NAME = 'ragdata'
RAGDATA_DIR = 'E:\\Docker\\RAGData'  # 中间文件存放目录
SCAN_INTERVAL = 120  # 扫描间隔（秒）
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

# 新增：token计数与分割工具
try:
    import tiktoken
except ImportError:
    tiktoken = None
    print('警告：未安装tiktoken库，token计数功能不可用。请pip install tiktoken')

def count_tokens(text, model_name='gpt-3.5-turbo'):
    if not tiktoken:
        return len(text)
    enc = tiktoken.encoding_for_model(model_name)
    return len(enc.encode(text))

def split_text_by_token(text, max_tokens=49000, model_name='gpt-3.5-turbo'):
    """
    按最大token数分割文本，优先按最后一个换行符分割。
    返回分片列表。
    """
    if not tiktoken:
        # 兜底：按字符粗略分割
        avg = max_tokens * 2  # 粗略估算
        return [text[i:i+avg] for i in range(0, len(text), avg)]
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(text)
    idx = 0
    result = []
    while idx < len(tokens):
        end = min(idx + max_tokens, len(tokens))
        chunk = enc.decode(tokens[idx:end])
        # 优先按最后一个换行符分割
        if end < len(tokens):
            last_nl = chunk.rfind('\n')
            if last_nl > 0:
                chunk = chunk[:last_nl+1]
                end = idx + len(enc.encode(chunk))
        result.append(chunk)
        idx = end
    return result

# 修改request_markdown_async，支持分片

def request_markdown_async(file_uid, text_path, chunk_idx=None):
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # 判断是否需要分片
    chunks = split_text_by_token(text, max_tokens=49000)
    for i, chunk in enumerate(chunks):
        payload = {'file_uid': f'{file_uid}_part{i+1}' if len(chunks)>1 else file_uid, 'text': chunk}
        try:
            requests.post(N8N_MARKDOWN_URL, json=payload, timeout=3)
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
    # 检查是否已存在该file_uid或其分片，存在则删除（全量覆盖）
    search_result = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.file_uid",
                    match=rest.MatchText(text=file_uid)
                )
            ]
        ),
        limit=1000
    )
    if search_result and search_result[0]:
        print(f"Qdrant已存在file_uid(含分片)={file_uid}，执行upsert...")
        ids = [point.id for point in search_result[0]]
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=rest.PointIdsList(points=ids)
        )
    vectorstore.add_documents(docs)
    print(f"已分块写入Qdrant向量库: {file_uid}，共{len(docs)}块")

def pdf_to_text(pdf_path, text_path, dir_id=None, file_uid=None):
    doc = fitz.open(pdf_path)
    text_content = ""
    # 计算图片保存目录
    if dir_id and file_uid:
        img_dir = os.path.join(RAGDATA_DIR, str(dir_id))
        ensure_dir(img_dir)
    else:
        img_dir = os.path.dirname(text_path)
    for i, page in enumerate(doc):
        print(f" - 正在处理第{i+1}页...")
        images = page.get_images(full=True)
        if images:
            ocr_text = ""
            for img_index, img in enumerate(images):
                xref = img[0]
                if dir_id and file_uid:
                    img_path = os.path.join(img_dir, f"{file_uid}_page{i+1}_img{img_index+1}.png")
                else:
                    img_path = os.path.join(img_dir, f"temp_page_{i+1}_img_{img_index+1}.png")
                # 如果图片已存在，直接用现有图片
                if not os.path.exists(img_path):
                    try:
                        pix = fitz.Pixmap(doc, xref)
                        if pix.n > 4:
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        pix.save(img_path)
                        pix = None
                    except Exception as e:
                        print(f"保存图片出错: {img_path}, 错误: {e}")
                        continue  # 跳过本图片
                # OCR识别，遇到错误跳过
                try:
                    ocr_text += pytesseract.image_to_string(img_path, lang='chi_sim+eng') + "\n"
                except Exception as e:
                    print(f"OCR图片出错: {img_path}, 错误: {e}")
                    continue  # 跳过本图片
            text_content += f"{ocr_text}\n"
        else:
            text = page.get_text()
            if text.strip():
                text_content += f"{text}\n"
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(text_content)

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
                            print(f'已请求n8n生成markdown（如超长将自动分片），等待回调: {fname}')
                            break
                        elif proc_state == ProcState.MARKDOWN:
                            print(f'即将写入向量库: {md_path}')
                            process_md_to_vector(md_path, base_name)
                            update_file_state(dir_id, fname, {'proc_state': ProcState.VECTOR, 'last_proc_time': now})
                            print(f'已写入向量库: {fname}')
                        elif proc_state == ProcState.VECTOR:
                            break  # 已是最终状态，不再处理
                        elif proc_state == ProcState.NEW and ext == '.pdf':
                            print(f'检测到PDF文件，直接抽取文本: {fname} -> {text_path}')
                            pdf_to_text(video_path, text_path, dir_id=dir_id, file_uid=base_name)
                            update_file_state(dir_id, fname, {'proc_state': ProcState.RAW_TEXT, 'last_proc_time': now})
                            print(f'PDF已抽取为原始文本: {fname}')
                            # 自动异步请求n8n生成markdown
                            request_markdown_async(base_name, text_path)
                            print(f'已请求n8n生成markdown（如超长将自动分片），等待回调: {fname}')
                            break
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
    # 判断是否分片
    is_part = '_part' in file_uid
    # 获取主file_uid
    main_file_uid = file_uid.split('_part')[0] if is_part else file_uid
    # 查找主文件的dir_id
    rec = dir_file_state_col.find_one({'file_uid': main_file_uid})
    if not rec:
        return jsonify({'status': 'fail', 'msg': '未找到file_uid'}), 404
    dir_id = rec['dir_id']
    rag_dir = os.path.join(RAGDATA_DIR, dir_id)
    if not os.path.exists(rag_dir):
        os.makedirs(rag_dir)
    md_path = os.path.join(rag_dir, file_uid + '.md')
    # 保存markdown
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(text)
    # 分片直接写入向量库
    process_md_to_vector(md_path, file_uid)
    print(f'收到n8n回调并保存markdown: {md_path}，已写入向量库')
    # 非分片才推进主文件状态
    if not is_part:
        update_file_state(dir_id, rec['file_name'], {'proc_state': ProcState.MARKDOWN, 'last_proc_time': int(time.time())})
    return jsonify({'status': 'ok'})

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
    resp_json = resp.json()
    if "tenant_access_token" not in resp_json:
        print("获取飞书token失败，返回内容：", resp_json)
        raise Exception("飞书token获取失败，请检查app_id/app_secret和网络")
    token = resp_json.get("tenant_access_token")
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
