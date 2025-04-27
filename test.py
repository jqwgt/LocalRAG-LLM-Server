import requests
import json
import time

# 定义 API 服务器地址
API_URL = "http://localhost:5000"  # 替换为实际的 API 地址

# 全局参数，用户可在此修改参数。当参数为 None 时，每次提问前会交互式输入
token = "your_token"   # 建议在这里填入 避免每次手动输入
model = None            # 要使用的模型 默认 None
enable_search = None    # 是否使用联网默认 None
use_vector = None       # 是否使用RAG，默认 None
# 查询任务状态
def get_task_status(task_id, token=token):
    """查询任务状态"""
    try:
        response = requests.get(f"{API_URL}/status/{task_id}", params={"token": token})
        if response.status_code == 200:
            return response.json()
        else:
            print(f"获取任务状态失败: HTTP {response.status_code}")
            return None
    except Exception as e:
        print("获取任务状态时出错:", str(e))
        return None

# 发送查询请求
def query_api(user_input, retries=3):
    """
    发送查询任务，并轮询任务状态。
    如果全局参数 token、model、enable_search 和 use_vector 为 None，则向用户交互式输入，
    否则直接使用全局预定义的值。
    :param user_input: 用户输入的问题
    :param retries: 失败时的最大重试次数
    """
    global token, model, enable_search, use_vector

    # 如果 token 为空，则交互式输入（可修改脚本前的默认值覆盖）
    if token is None:
        token = input("请输入用户 token: ").strip()
    # 如果 model 为空，则交互式输入
    if model is None:
        model = input("请输入模型（例如 qwen2.5:72b）: ").strip() or "qwen2.5:72b"
    # 如果 enable_search 为空，则交互式输入
    if enable_search is None:
        enable_search_input = input("是否使用联网搜索? (y/n): ").strip().lower()
        enable_search = True if enable_search_input in ("y", "yes") else False
    # 如果 use_vector 为空，则交互式输入
    if use_vector is None:
        use_vector_input = input("是否使用本地向量知识库检索? (y/n): ").strip().lower()
        use_vector = True if use_vector_input in ("y", "yes") else False

    for attempt in range(retries):
        try:
            # 发送任务请求
            payload = {
                "token": token,
                "model": model,
                "input": user_input,
                "enable_search": enable_search,
                "use_vector": use_vector
            }
            response = requests.post(
                f"{API_URL}/query",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=10  # 超时 10 秒
            )

            if response.status_code == 200:
                task_id_resp = response.json().get("task_id")
                print(f"任务已提交，任务 ID: {task_id_resp}")

                # 轮询任务状态
                while True:
                    status_data = get_task_status(task_id_resp, token)
                    if status_data:
                        status = status_data.get("status")
                        if status == "完成":
                            print("任务完成，结果:", status_data.get("result"))
                            return
                        elif status == "失败":
                            print("任务失败，错误信息:", status_data.get("error"))
                            return
                        else:
                            print("任务处理中...")
                    else:
                        print("获取任务状态失败")
                        break
                    time.sleep(5)  # 每 5 秒轮询一次
            else:
                print(f"错误: HTTP {response.status_code}")
                print("响应内容:", response.json())
        except requests.exceptions.Timeout:
            print(f"请求超时，重试中 ({attempt + 1}/{retries})...")
        except Exception as e:
            print("发生错误:", str(e))
            break
    print("请求失败，请稍后重试。")

# 主程序
if __name__ == "__main__":
    print("测试 Flask API 接口")
    while True:
        user_question = input("请输入您的问题 (输入 '退出' 结束测试): ")
        if user_question.lower() == '退出':
            print("测试结束。")
            break
        query_api(user_question)
