import warnings
import os
import subprocess
import time
import uuid
import logging
import requests
from datetime import datetime
from threading import Thread

from flask import Flask, request, jsonify, redirect, url_for, flash, render_template
from flask_cors import CORS

# 新增：数据库和管理界面
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView

# 导入原有模型相关库（假设这些库均已安装）
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAIimport warnings
import os
import subprocess
import time
import uuid
import logging
import requests
from datetime import datetime
from threading import Thread

from flask import Flask, request, jsonify, redirect, url_for, flash, render_template
from flask_cors import CORS

# 新增：数据库和管理界面
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView

# 导入原有模型相关库（假设这些库均已安装）
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAIimport warnings
import os
import subprocess
import time
import uuid
import logging
import requests
from datetime import datetime
from threading import Thread

from flask import Flask, request, jsonify, redirect, url_for, flash, render_template
from flask_cors import CORS

# 新增：数据库和管理界面
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView

# 导入原有模型相关库（假设这些库均已安装）
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.agents import initialize_agent, AgentType
from wtforms.validators import DataRequired

# 用户认证相关库
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from os import environ
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from flask_admin import Admin, AdminIndexView

# 新增 DuckDuckGo 搜索工具（基于 langchain 工具接口）
from langchain.tools.base import BaseTool

warnings.filterwarnings("ignore", category=DeprecationWarning)

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app)  # 允许所有跨域请求

# 配置数据库（SQLite 示例）
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# 配置 Flask-Admin需要的 secret key
app.config['SECRET_KEY'] = 'your-secret-key'
db = SQLAlchemy(app)


# 配置项
class Config:
    SECRET_KEY = environ.get('SECRET_KEY') or 'dev-key-please-change'
    ADMIN_USERNAME = environ.get('ADMIN_USERNAME') or 'admin'
    ADMIN_PASSWORD = environ.get('ADMIN_PASSWORD') or 'admin'


app.config.from_object(Config)


# ---------------------------
# 定义数据模型
# ---------------------------
# 修改 User 模型，新增 name 字段
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128))  # 新增的可选名称/备注字段
    token = db.Column(db.String(128), unique=True, nullable=False)
    user_type = db.Column(db.String(16), default="normal")  # normal 或 admin
    first_ip = db.Column(db.String(64))
    first_device = db.Column(db.String(128))
    active = db.Column(db.Boolean, default=True)

    usage_logs = db.relationship("UsageLog", back_populates="user", lazy=True)

    def __repr__(self):
        return f"<User {self.id} - {self.name or ''} - {self.user_type} - {'Active' if self.active else 'Suspended'}>"


# 修改 UsageLog 模型，添加 user_detail 属性
class UsageLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    ip = db.Column(db.String(64))
    question = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship("User", back_populates="usage_logs")

    @property
    def user_detail(self):
        if self.user:
            return f"{self.user.id} - {self.user.name or ''}"
        return ""

    def __repr__(self):
        return f"<UsageLog User:{self.user_id} - {self.timestamp}>"


# 管理员用户模型
class AdminUser(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


# 初始化数据库（第一次运行时会创建表）
with app.app_context():
    db.create_all()

# ---------------------------
# 在 Admin 初始化之前添加自定义 ModelView
# ---------------------------
from flask_admin.contrib.sqla import ModelView
from wtforms.validators import DataRequired


class UserModelView(ModelView):
    column_list = ('id', 'name', 'token', 'user_type', 'first_ip', 'first_device', 'active')
    form_columns = ('name', 'token', 'user_type', 'active')
    form_choices = {
        'user_type': [
            ('normal', '普通用户'),
            ('admin', '管理员')
        ]
    }
    form_args = {
        'token': {
            'validators': [DataRequired()],
            'description': '用户唯一标识'
        },
        'user_type': {
            'validators': [DataRequired()],
            'description': '用户类型'
        },
        'name': {
            'description': '可选的用户名/备注'
        }
    }

    def on_form_prefill(self, form, id):
        print("Form fields:", form._fields)
        print("Form choices for user_type:", form.user_type.choices)


# 修改 UsageLogModelView，支持根据 user_id 或 user.name 筛选
class UsageLogModelView(ModelView):
    column_list = ('id', 'user_detail', 'ip', 'question', 'timestamp')
    form_columns = ('user_id', 'ip', 'question')
    # 支持根据 user_id、user.name、ip 和 timestamp 筛选
    column_filters = ('user_id', 'user.name', 'ip', 'timestamp')


# 安全的 ModelView
class SecureModelView(ModelView):
    def is_accessible(self):
        return current_user.is_authenticated


# 初始化 LoginManager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return AdminUser.query.get(int(user_id))


# 单例 Admin 实例
class AdminView(Admin):
    def is_accessible(self):
        return current_user.is_authenticated

    def inaccessible_callback(self, name, **kwargs):
        return redirect(url_for('login', next=request.url))


# 创建单个 Admin 实例
admin = AdminView(
    app,
    name='管理后台',
    template_mode='bootstrap4',
    index_view=AdminIndexView(
        name='首页',
        template='admin/index.html',
        url='/admin'
    )
)

# 添加视图
admin.add_view(UserModelView(User, db.session, name='用户管理'))
admin.add_view(UsageLogModelView(UsageLog, db.session, name='使用记录'))


# 创建初始管理员账号
def init_admin():
    if not AdminUser.query.filter_by(username=Config.ADMIN_USERNAME).first():
        admin_user = AdminUser(username=Config.ADMIN_USERNAME)
        admin_user.set_password(Config.ADMIN_PASSWORD)
        db.session.add(admin_user)
        db.session.commit()


# 在应用启动时初始化
with app.app_context():
    db.create_all()
    init_admin()

# ---------------------------
# 全局配置变量（AI 模型相关）
# ---------------------------
# 默认温度参数与调用速率控制
MODEL_TEMPERATURE = 0.7
MAX_TOKENS_PER_SECOND = 10
KNOWLEDGE_BASE_PATH = "docs/knowledge_base.txt"  # 知识库文件路径
EMBEDDING_MODEL_PATH = "./model"  # 嵌入模型路径
OLLAMA_SERVICE_STARTUP_DELAY = 5  # Ollama 服务启动等待时间（秒）

# 支持的 Ollama 大模型列表（同时也是允许用户选择的模型）
AVAILABLE_MODELS = ["deepseek-r1:70b", "qwen2.5:72b", "glm4:latest"]

# 存储不同组合的 LangChain 组件,key 为 (model, use_search)
langchain_sets = {}

# 日志配置（控制台日志），确保终端可实时输出
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------
# 新增 DuckDuckGo 搜索工具（免费 API）
# ---------------------------
class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGoSearch"
    description: str = "使用 DuckDuckGo 搜索 API 进行信息检索。输入应为查询字符串。"

    def _run(self, query: str):
        logger.info(f"调用 DuckDuckGo 搜索，查询内容：{query}")
        try:
            resp = requests.get("https://api.duckduckgo.com", params={"q": query, "format": "json", "no_redirect": 1},
                                timeout=5)
            return resp.json()
        except Exception as e:
            logger.error(f"DuckDuckGo 搜索失败: {e}")
            return {"error": str(e)}

    async def _arun(self, query: str):
        raise NotImplementedError("DuckDuckGoSearchTool 不支持异步调用")


# ---------------------------
# 检查并启动 Ollama 服务（支持多个模型）
# ---------------------------
def check_and_start_ollama():
    # 获取当前已启动模型列表
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        running_models = result.stdout if result.returncode == 0 else ""
    except Exception as e:
        logger.error(f"检查 Ollama 服务时出错: {e}")
        running_models = ""
    # 对每个模型进行检查并使用 --verbose 模式启动
    for model in AVAILABLE_MODELS:
        if model in running_models:
            logger.info(f"Ollama 服务已启动：{model}")
        else:
            logger.info(f"Ollama 服务未启动：{model}，正在尝试启动...")
            try:
                subprocess.Popen(["ollama", "run", "--verbose", model],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
                time.sleep(OLLAMA_SERVICE_STARTUP_DELAY)
                logger.info(f"Ollama 服务启动完成：{model}")
            except Exception as e:
                logger.error(f"启动 Ollama 服务 {model} 失败: {e}")


# ---------------------------
# 加载本地知识库并构建向量数据库
# ---------------------------
def setup_knowledge_base():
    logger.info("正在加载本地知识库并构建向量数据库...")
    loader = TextLoader(KNOWLEDGE_BASE_PATH, encoding="utf-8")
    documents = loader.load()
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
    texts = [doc.page_content for doc in documents]
    vectorstore = FAISS.from_texts(texts, embedding_model)
    logger.info("向量数据库构建完成！")
    return vectorstore


# 全局向量数据库
vectorstore = setup_knowledge_base()
# 定义一个 DummyRetriever，如果不使用本地向量知识库，则返回空列表
from langchain.schema import BaseRetriever


# 定义一个 DummyRetriever，如果不使用本地向量知识库，则返回空列表
class DummyRetriever(BaseRetriever):
    def get_relevant_documents(self, query: str):
        return []

    async def aget_relevant_documents(self, query: str):
        return self.get_relevant_documents(query)

    @property
    def input_keys(self):
        return []

    @property
    def output_keys(self):
        return []


# ---------------------------
# 初始化 LangChain 模型和检索式问答（支持是否联网搜索）
# ---------------------------
# 修改初始化 LangChain 模型函数，增加 use_vector 参数
def setup_langchain_for_model(model, use_search, use_vector=True):
    logger.info(f"正在初始化 LangChain 模型，模型：{model}，联网搜索：{use_search}，使用向量检索：{use_vector}")
    llm = OllamaLLM(model=model, temperature=MODEL_TEMPERATURE, max_tokens_per_second=MAX_TOKENS_PER_SECOND)
    # 如果启用搜索，则添加 DuckDuckGo 搜索工具，否则工具列表置空
    if use_search:
        tools = [DuckDuckGoSearchTool()]
    else:
        tools = []
    if tools:
        agent_executor = initialize_agent(
            tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True
        )
    else:
        agent_executor = None
    # 根据 use_vector 决定使用真实向量库还是 DummyRetriever
    if use_vector:
        retriever = vectorstore.as_retriever()
    else:
        retriever = DummyRetriever()
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    logger.info("LangChain 初始化完成！")
    return retrieval_qa, agent_executor


# 扩展预创建实例，新增 use_vector 维度
for model in AVAILABLE_MODELS:
    for search_enabled in [False, True]:
        for vector_enabled in [False, True]:
            langchain_sets[(model, search_enabled, vector_enabled)] = setup_langchain_for_model(model, search_enabled,
                                                                                                vector_enabled)

# 全局任务字典，用于存储任务状态
tasks = {}
# ---------------------------
# 处理任务（异步执行 AI 模型调用）
# ---------------------------
# 修改任务处理函数，接收 use_vector 参数
import re


def count_tokens(text: str) -> int:
    """
    简单统计 token 数：对中文，将每个汉字计为一个 token，
    对英文及数字则以单词计算。
    """
    # 匹配中文汉字或英文单词（包括下划线）
    tokens = re.findall(r"[\u4e00-\u9fff]|[\w']+", text)
    return len(tokens)


def process_task(task_id, user_input, model_type, use_search, use_vector):
    try:
        start_time = datetime.now()
        logger.info(f"开始处理任务 {task_id}，模型：{model_type}，联网搜索：{use_search}，使用向量检索：{use_vector}")
        # 从预创建实例中取出对应组合
        qa_chain, agent_executor = langchain_sets.get((model_type, use_search, use_vector))
        if qa_chain is None:
            raise ValueError(f"未找到对应的模型实例: {model_type}，搜索选项: {use_search}，向量检索: {use_vector}")

        # 调用检索问答链
        logger.info(f"调用检索问答链，用户输入：{user_input}")
        response = qa_chain({"query": user_input})
        logger.info(f"检索问答链返回结果：{response}")

        # 调用 Agent 链，仅在启用联网搜索时调用
        if agent_executor is not None:
            logger.info(f"调用 Agent 链，用户输入：{user_input}")
            agent_response = agent_executor.invoke({"input": user_input})
            logger.info(f"Agent 链返回结果：{agent_response}")
        else:
            logger.info("未启用联网搜索，跳过 Agent 链调用")
            agent_response = None

        end_time = datetime.now()
        # 推理时间，单位：秒
        inference_time = (end_time - start_time).total_seconds()
        # 使用自定义函数统计 token 数
        result_text = response.get("result", "")
        token_count = count_tokens(result_text)
        tokens_per_second = token_count / inference_time if inference_time > 0 else token_count

        performance_info = {
            "inference_time_seconds": inference_time,
            "token_count": token_count,
            "tokens_per_second": tokens_per_second
        }

        tasks[task_id] = {
            "status": "完成",
            "result": {
                "answer": result_text,
                "sources": [
                    {"content": doc.page_content, "metadata": doc.metadata}
                    for doc in response.get("source_documents", [])
                ],
                "agent_response": agent_response,
                "performance": performance_info
            }
        }
        logger.info(
            f"任务 {task_id} 处理完成，推理时间 {inference_time} 秒, token 数 {token_count}, tokens/s {tokens_per_second:.2f}")
    except Exception as e:
        tasks[task_id] = {"status": "失败", "error": str(e)}
        logger.error(f"任务 {task_id} 处理失败: {e}")


# 检查并启动所有 Ollama 服务
check_and_start_ollama()


# ---------------------------
# API 路由：用户请求查询接口
# ---------------------------
# 修改 /query 路由，支持指定模型以及是否启用联网搜索（参数 model 和 enable_search）
# 修改 /query API 路由，增加 use_vector 参数（默认 True）
@app.route('/query', methods=['POST'])
def query_model():
    try:
        data = request.json
        token = data.get("token")
        model_type = data.get("model")
        user_input = data.get("input", "")
        # 新增：是否启用联网搜索和向量检索，默认值分别为 False 和 True
        use_search = data.get("enable_search", False)
        use_vector = data.get("use_vector", True)

        if not token or not model_type or not user_input:
            return jsonify({"error": "缺少 token/model/input 参数"}), 400

        if model_type not in AVAILABLE_MODELS:
            return jsonify({"error": f"不支持的模型类型，仅支持：{AVAILABLE_MODELS}"}), 400

        user = User.query.filter_by(token=token).first()
        if not user:
            return jsonify({"error": "用户不存在或 token 错误"}), 401

        if not user.active:
            return jsonify({"error": "该用户已被暂停使用"}), 403

        device = request.headers.get('User-Agent') or 'unknown'
        current_ip = request.remote_addr
        if user.user_type == "normal":
            if not user.first_ip or not user.first_device:
                user.first_ip = current_ip
                user.first_device = device
                db.session.commit()
            else:
                if user.first_ip != current_ip or user.first_device != device:
                    user.active = False
                    db.session.commit()
                    return jsonify({"error": "账号异常使用，已暂停"}), 403

        usage = UsageLog(user_id=user.id, ip=current_ip, question=user_input, timestamp=datetime.utcnow())
        db.session.add(usage)
        db.session.commit()

        task_id = str(uuid.uuid4())
        tasks[task_id] = {"status": "处理中"}

        # 启动后台线程时传递 use_vector 参数
        thread = Thread(target=process_task, args=(task_id, user_input, model_type, use_search, use_vector))
        thread.start()

        return jsonify({"task_id": task_id, "status": "任务已接收"})
    except Exception as e:
        logger.error(f"/query 接口异常: {e}")
        return jsonify({"error": str(e)}), 500


# API 路由：查询任务状态
@app.route('/status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "任务不存在"}), 404
    return jsonify(task)


# ---------------------------
# 用户认证相关视图
# ---------------------------
# 登录表单
class LoginForm(FlaskForm):
    username = StringField('用户名', validators=[DataRequired()])
    password = PasswordField('密码', validators=[DataRequired()])
    submit = SubmitField('登录')


# 登录视图，修改为使用 AdminUser 模型进行验证
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        admin_user = AdminUser.query.filter_by(username=form.username.data).first()
        if admin_user and admin_user.check_password(form.password.data):
            login_user(admin_user)
            return redirect(url_for('admin.index'))
        flash('用户名或密码错误')
    return render_template('login.html', form=form)


# 登出视图
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# ---------------------------
# 主函数，启动 Flask 服务
# ---------------------------
if __name__ == "__main__":
    # 运行在 0.0.0.0，端口 5000，调试模式关闭（生产环境请使用 WSGI 部署）
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False, threaded=True)