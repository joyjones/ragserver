# 工程简介

本项目是一个本地资料自动化处理与智能问答系统，支持多种格式文件（如视频、音频、文档等）的自动化处理、文本抽取、向量化和智能问答。

## 核心功能
- 监控指定目录下的资料文件，自动检测新增或变更文件。
- 支持视频转音频（ffmpeg）、音频转文本（whisper）、文本转markdown（n8n+大模型）、markdown转向量（OpenAI Embedding + Qdrant）。
- 处理流程全自动，状态实时记录于MongoDB。
- 支持通过API进行智能问答，基于LangChain与本地向量库检索。
- 集成飞书机器人，支持在飞书群聊中直接提问并获得基于本地资料的智能回答。

## 技术栈
- Flask/FastAPI 提供API服务
- MongoDB 记录资料目录与文件处理状态
- ffmpeg/whisper/n8n 负责多模态文件处理与文本生成
- OpenAI Embedding + Qdrant 负责文本向量化与检索
- LangChain 负责智能问答
- 飞书开放平台对接群聊问答

## 主要流程
1. 定时扫描资料目录，发现新文件或变更文件。
2. 按需依次完成视频转音频、音频转文本、文本转markdown、markdown转向量等处理。
3. 处理状态与中间文件实时记录，便于追踪与恢复。
4. 用户可通过API或飞书机器人进行基于本地资料的智能问答。

---

## 预定义的处理目录
- **资料目录**：用户自定义的本地资料目录，支持多种文件格式（视频、音频、文本、各类软件文件等）。
- **RAGData**：本地用于存放所有处理中间文件的目录（如音频、文本等）。

## 数据库结构（MongoDB）
- 使用 `ragdata` 数据库，包含以下集合：

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

## 本地监控处理服务（analyser_server.py）
- 启动后每分钟定时检测，遍历所有目录和文件。
- 新文件自动记录，变更文件自动更新。
- 处理流程：
  1. 原始视频 → 音频
  2. 音频 → 原始文本
  3. 原始文本 → markdown（调用n8n+大模型）
  4. markdown → 向量数据
  5. xlsx/docx/pptx等文档也可直接转markdown
- 过程文件存储于RAGData目录，处理状态实时写入MongoDB。

## 依赖的工具
- **ffmpeg**：视频转音频
- **whisper**：音频转文本
- **n8n**：文本转markdown、文档转markdown