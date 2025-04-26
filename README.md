# LocalRAG-LLM-Server

![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)

LocalRAG-LLM-Server 是一个基于 Flask、LangChain 和 Ollama 的本地 RAG（Retrieval-Augmented Generation）后端服务，集成多模型管理、管理后台及远程 API，支持本地向量检索与外部联网搜索。

## 特性

- **多模型管理**：自动检查并启动 Ollama 本地模型（`deepseek-r1:70b`、`qwen2.5:72b`、`qwq:latest`等）。
- **RAG 系统**：使用 MiniLM-L6-H384-uncased（配置在 `model/config.json`）构建本地向量数据库检索。
- **LangChain 集成**：支持 RetrievalQA、AgentType.ZERO_SHOT_REACT_DESCRIPTION 以及自定义 DuckDuckGo 搜索工具。
- **Flask 后端**：提供 `/query` 和 `/status/<task_id>` RESTful API 接口。
- **管理后台**：基于 Flask-Admin，支持用户与使用记录管理。
- **开源协议**：使用 GPL-3.0 开源协议发布。

## 目录结构

```
.
├── app.py                  # Flask 服务及业务逻辑
├── test.py                 # API 测试客户端
├── requirements.txt        # Python 依赖
├── README.md               # 项目说明
├── docs/
│   └── knowledge_base.txt  # 本地检索知识库
└── model/
    ├── config.json         # MiniLM-L6 模型配置
    ├── pytorch_model.bin   # 模型权重
    └── vocab.txt           # 词表
```

## 安装与运行

1. **克隆仓库，并进入目录**

   ```bash
   git clone https://github.com/jqwgt/LocalRAG-LLM-Server.git
   cd LocalRAG-LLM-Server
   ```

2. **下载 MiniLM-L6-H384-uncased 模型**

   - **推荐来源**：  
     - **官方 HuggingFace**：[https://huggingface.co/sentence-transformers/mini-lm-l6-v2-h384-uncased](https://huggingface.co/sentence-transformers/mini-lm-l6-v2-h384-uncased)  
     - **国内镜像站**（加速下载）：[https://hf-mirror.com/sentence-transformers/mini-lm-l6-v2-h384-uncased](https://hf-mirror.com/sentence-transformers/mini-lm-l6-v2-h384-uncased)  
   - **下载文件**：  
     需下载以下 3 个核心文件（总大小约 200MB）：  
     ```
     config.json
     pytorch_model.bin
     vocab.txt
     ```
3. **安装依赖**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **配置环境变量或在 app.py 中设定（推荐使用环境变量）**

   - `SECRET_KEY`：Flask 会话密钥
   - `ADMIN_USERNAME` / `ADMIN_PASSWORD`：管理员登录凭据

5. **安装并启动 Ollama**

   ```bash
   brew install ollama
   ollama pull deepseek-r1:70b ...
   ```

6. **放置知识库文档**

   将检索文档放在 `docs/knowledge_base.txt` 中，每行一条或一个大文本。

7. **启动服务**

   ```bash
   python app.py
   ```

8. **访问管理后台**

   打开浏览器访问 `http://localhost:5000/admin`，使用管理员账号登录。

9. **调用 API**

   - **提交查询**：

     ```
     POST /query
     Content-Type: application/json

     {
       "token": "<user-token>",
       "model": "qwen2.5:72b",
       "input": "你的问题",
       "enable_search": false,
       "use_vector": true
     }
     ```

   - **查询状态**：

     ```
     GET /status/<task_id>?token=<user-token>
     ```

10. **使用示例客户端**

   ```bash
   python test.py
   ```

## 模型与检索

- **嵌入模型**：`model/config.json` 中 MiniLM-L6-H384-uncased，用于 FAISS 向量检索。
- **网络搜索**：可选 DuckDuckGo 免费 API 或者其他 API。

## 授权协议

本项目采用 GPL-3.0 License 发布，详情见 [LICENSE](LICENSE)。
