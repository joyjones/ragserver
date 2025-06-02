# 本地RAG智能知识库搭建与使用指南

## 一、项目简介

本项目是一个本地资料自动化处理与智能问答系统，支持多种格式文件（如视频、音频、文档等）的自动化处理、文本抽取、向量化和智能问答。系统可自动监控指定目录，处理新文件，并通过API或飞书机器人实现基于本地资料的智能问答。

---

## 二、核心功能

- 自动监控资料目录，检测新增或变更文件。
- 多模态处理：支持视频转音频（ffmpeg）、音频转文本（whisper）、文本转markdown（n8n+大模型）、markdown转向量（OpenAI Embedding + Qdrant）。
- 全流程自动化，处理状态实时记录于MongoDB。
- API智能问答，基于LangChain与本地向量库检索。
- 飞书机器人集成，支持群聊中直接提问并获得基于本地资料的智能回答。

---

## 三、技术栈

- Flask 提供API服务
- MongoDB 记录资料目录与文件处理状态
- ffmpeg/whisper/n8n 负责多模态文件处理与文本生成
- OpenAI Embedding + Qdrant 负责文本向量化与检索
- LangChain 负责智能问答
- 飞书开放平台对接群聊问答
- Docker Compose 一键部署全部依赖服务

---

## 四、环境准备

### 1. Python依赖

请确保已安装 Python 3.8+，并安装以下依赖（建议使用虚拟环境）：

```bash
pip install flask pymongo langchain-openai langchain-community qdrant-client fastapi pydantic pdfplumber pymupdf pytesseract pdf2image requests
```

如需支持token分割，建议安装`tiktoken`：

```bash
pip install tiktoken
```

### 2. 系统依赖

- **ffmpeg**：用于视频转音频。  
  [官方下载地址](https://ffmpeg.org/download.html)  
  安装后请确保`ffmpeg`命令可在系统PATH中直接调用。

- **whisper**：用于音频转文本。  
  推荐使用 [openai/whisper](https://github.com/openai/whisper) 官方实现，需安装PyTorch环境。

- **tesseract**：用于图片型PDF的OCR识别。  
  [官方下载地址](https://github.com/tesseract-ocr/tesseract/releases)  
  安装后请确保`tesseract`命令可在系统PATH中直接调用。  
  **如需识别中文（简体），请务必安装 `chi_sim` 语言包。**

- **n8n**：用于文本转markdown、文档转markdown。  
  推荐使用docker-compose一键部署。

---

## 五、Docker一键部署

项目已提供 `docker-compose.yml`，可一键部署以下服务：

- Qdrant（向量数据库）
- n8n（自动化流程引擎）
- MongoDB（数据库）
- mongo-express（MongoDB可视化管理）
- langchain（可选，需自定义API）

### 步骤

1. 进入 `docker-compose.yml` 所在目录（如 `e:\Docker`）。
2. 执行：

   ```bash
   docker-compose up -d
   ```

3. 各服务端口说明：
   - Qdrant: 6333
   - n8n: 5678
   - MongoDB: 27017
   - mongo-express: 8081
   - langchain: 8000

---

## 六、目录结构与配置

- **资料目录**：用户自定义的本地资料目录，支持多种文件格式（如视频、音频、文本等）。
- **RAGData**：本地用于存放所有处理中间文件的目录（如音频、文本等），默认路径为 `E:\Docker\RAGData`，可在 `app.py` 中修改。

### MongoDB结构

- `dir_index`：记录需要分析的文件目录。
- `dir_file_state`：记录每个目录下每个文件的最新处理状态。

---

## 七、数据结构说明

### 1. `dir_index` 表
记录本地需要分析的文件目录。
字段示例：
```json
{
  "dirPath": "F:\\课程\\杨天真的32个高情商公式",
  "recursive": false,
  "fileTypes": [".mp4"],
  "enabled": true,
  "lastScanTime": null,
  "notes": "杨天真的32个高情商公式"
}
```

### 2. `dir_file_state` 表
记录每个目录下每个文件的最新处理状态。
字段示例：
```json
{
  "dir_id": "xxxx",
  "file_name": "xxxx.mp4",
  "file_size": 1234,
  "file_type": 4,
  "file_md5": "",
  "proc_state": 0,
  "last_proc_time": 0,
  "last_modify_time": 0
}
```

- **proc_state 定义：**
  - 0：新增文件（新发现的文件，加入到记录）
  - 1：已处理为音频（如视频转音频）
  - 2：已处理为图片（如视频/PPT提取图片）
  - 3：已处理为原始文本（音频转文本，未加工）
  - 4：已处理为markdown可读文本（大模型/插件处理后）
  - 5：已生成向量数据（文本/多模态数据导入向量库）

---

## 八、环境变量配置

- `OPENAI_API_KEY`：用于OpenAI Embedding和LangChain问答，需提前在环境变量中设置。
- `FEISHU_APP_ID`、`FEISHU_APP_SECRET`：飞书机器人相关配置，可在环境变量中设置，也可直接在`app.py`中修改默认值。

---

## 九、启动主服务

1. 启动依赖服务（见上文docker-compose）。
2. 启动主服务：

   ```bash
   python app.py
   ```

   启动后会自动进入资料目录监控与处理主循环，并开放API服务（默认端口9927）。

---

## 十、API接口说明

### 1. 测试接口

- `GET /test`  
  用于测试服务是否正常。

### 2. 智能问答接口

- `POST /ask`  
  请求体示例：
  ```json
  {
    "question": "你的问题内容",
    "top_k": 3
  }
  ```
  返回：
  ```json
  {
    "answer": "智能回答内容",
    "context": "检索到的资料片段"
  }
  ```

### 3. 保存markdown回调

- `POST /save_markdown`  
  供n8n回调，自动保存markdown并写入向量库。

### 4. 飞书机器人事件

- `POST /feishu/event`  
  飞书开放平台事件回调接口，支持群聊智能问答。

---

## 十一、常见问题与注意事项

- **tesseract中文识别**：务必安装`chi_sim`语言包，并配置好环境变量。
- **whisper模型**：默认使用`large-v3`，如需更换请在`app.py`中修改。
- **n8n流程**：需自行配置n8n的markdown转换webhook，地址需与`app.py`中一致。
- **环境变量**：如未设置`OPENAI_API_KEY`等，相关功能将无法使用。
- **目录权限**：确保`RAGData`等目录有读写权限。

---

## 十二、进阶用法

- 支持多目录、多类型文件自动处理。
- 支持分片处理超长文本，自动分片写入向量库。
- 支持自定义处理流程与扩展。

---

## 十三、参考与致谢

- [LangChain](https://github.com/langchain-ai/langchain)
- [Qdrant](https://qdrant.tech/)
- [n8n](https://n8n.io/)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

---

如有问题欢迎提issue或联系作者。