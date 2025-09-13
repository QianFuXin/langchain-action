"""
自动原生实现工具调用、历史会话存储等功能
"""
import os
import time
import json
import pickle
import inspect
import typing
import logging
from pathlib import Path
from openai import OpenAI
from config import *

# =========================
# 日志配置
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("function_call_playground")

# =========================
# OpenAI 客户端配置
# =========================
client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")


# =========================
# 工具装饰器
# =========================
def make_tool_decorator(registry: list, func_map: dict):
    TYPE_MAP = {int: "number", float: "number", str: "string", bool: "boolean"}

    def tool(description: str = ""):
        def wrapper(func):
            sig = inspect.signature(func)
            hints = typing.get_type_hints(func)

            properties, required = {}, []
            for name, param in sig.parameters.items():
                typ = hints.get(name, str)
                json_type = TYPE_MAP.get(typ, "string")
                properties[name] = {"type": json_type}
                required.append(name)

            schema = {
                "type": "function",
                "function": {
                    "name": func.__name__,
                    "description": description or (func.__doc__ or "").strip(),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
            registry.append(schema)
            func_map[func.__name__] = func
            logger.debug(f"🔧 注册工具: {func.__name__} -> {schema}")
            return func

        return wrapper

    return tool


# =========================
# 工具函数注册
# =========================
my_tools: list = []
func_registry: dict = {}
tool = make_tool_decorator(my_tools, func_registry)


@tool("Compute the sum of two numbers")
def add(a: float, b: float) -> float:
    return a + b


@tool("Calculate the product of two numbers")
def mul(a: float, b: float) -> float:
    return a * b


@tool("Compare two numbers, which one is bigger")
def compare(a: float, b: float) -> str:
    if a > b:
        return f"{a} is greater than {b}"
    elif a < b:
        return f"{b} is greater than {a}"
    return f"{a} is equal to {b}"


# =========================
# 会话存储
# =========================
DEFAULT_STORE = Path(os.environ.get("CHAT_SESSION_DIR", Path.home() / ".chat_sessions"))
DEFAULT_STORE.mkdir(parents=True, exist_ok=True)
STORE_DIR = DEFAULT_STORE


def load_messages(session_id: str) -> list:
    """加载历史对话"""
    path = STORE_DIR / f"{session_id}.pkl"
    if path.exists():
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
                logger.debug(f"📂 加载历史对话: {session_id}, 条数={len(data)}")
                return data
        except Exception as e:
            logger.warning(f"⚠️ 会话文件损坏，已忽略: {path}, error={e}")
            return []
    return []


def save_messages(session_id: str, messages: list) -> None:
    """保存对话"""
    path = STORE_DIR / f"{session_id}.pkl"
    try:
        with open(path, "wb") as f:
            pickle.dump(messages, f)
        logger.debug(f"💾 已保存会话: {session_id}, 条数={len(messages)}")
    except Exception as e:
        logger.error(f"❌ 保存会话失败: {path}, error={e}")


# =========================
# 主逻辑
# =========================
def function_call_playground(
        prompt: str,
        tools=None,
        session_id: str = None,
        system_prompt: str = "You are a helpful AI assistant.",
) -> str:
    """主对话逻辑"""

    session_id = session_id or str(int(time.time()))
    messages = load_messages(session_id)

    # 仅在新会话时添加 system prompt
    if not any(msg["role"] == "system" for msg in messages):
        system_msg = {"role": "system", "content": system_prompt}
        messages.insert(0, system_msg)
        logger.info(f"🛠️ 使用 system prompt: {system_prompt}")

    # 当前用户输入
    messages.append({"role": "user", "content": prompt})
    while True:
        response = client.chat.completions.create(
            model="Qwen/Qwen3-14B",
            messages=messages,
            temperature=0.01,
            top_p=0.95,
            stream=False,
            tools=tools,
        )

        msg = response.choices[0].message

        # 模型直接回答
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content})
            save_messages(session_id, messages)
            logger.info(f"🤖 模型回复: {msg.content}")
            return msg.content

        # 模型调用工具
        messages.append(msg.to_dict())

        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            logger.info(f"👉 模型请求调用函数: {func_name}({func_args})")

            if func_name not in func_registry:
                result = f"❌ Unknown function: {func_name}"
            else:
                try:
                    result = func_registry[func_name](**func_args)
                except Exception as e:
                    result = f"❌ Error: {e}"

            logger.info(f"🔧 执行结果: {result}")

            tool_msg = {
                "role": "tool",
                "content": str(result),
                "tool_call_id": tool_call.id,
            }
            messages.append(tool_msg)

        save_messages(session_id, messages)


print(
    function_call_playground(
        "请计算 12.34 + 56.78 的值", tools=my_tools, session_id="test_session_001"
    )
)

print(
    function_call_playground(
        "我说的上句话是什么？", tools=my_tools, session_id="test_session_001"
    ))
