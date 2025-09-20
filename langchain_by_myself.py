"""
支持本地工具 + 远程 MCP 工具调用
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

import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

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


def parse_tool_result(result):
    """统一解析 MCP 工具调用结果"""
    if hasattr(result, "structuredContent") and result.structuredContent:
        # 结构化优先
        return result.structuredContent.get("result")
    if hasattr(result, "content") and result.content:
        # 文本结果
        return getattr(result.content[0], "text", None)
    return None


# =========================
# 本地工具注册
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
            logger.debug(f"🔧 注册本地工具: {func.__name__}")
            return func

        return wrapper

    return tool


my_tools: list = []
func_registry: dict = {}
tool = make_tool_decorator(my_tools, func_registry)


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
    path = STORE_DIR / f"{session_id}.pkl"
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"⚠️ 会话文件损坏: {path}, error={e}")
            return []
    return []


def save_messages(session_id: str, messages: list) -> None:
    path = STORE_DIR / f"{session_id}.pkl"
    try:
        with open(path, "wb") as f:
            pickle.dump(messages, f)
    except Exception as e:
        logger.error(f"❌ 保存会话失败: {path}, error={e}")


# =========================
# MCP 工具支持
# =========================
def tool_to_openai_format(tool):
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": {
                "type": "object",
                "properties": {
                    k: {
                        "type": "number" if v.get("type") in ("integer", "number") else v.get("type", "string")
                    }
                    for k, v in tool.inputSchema.get("properties", {}).items()
                },
                "required": tool.inputSchema.get("required", []),
            },
        },
    }


async def fetch_mcp_tools():
    async with streamablehttp_client("http://localhost:8001/mcp") as (
            read_stream,
            write_stream,
            _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()
            tools = result.tools if hasattr(result, "tools") else result[-1]
            return [tool_to_openai_format(tool) for tool in tools]


def get_mcp_tools():
    return asyncio.run(fetch_mcp_tools())


async def call_mcp_tool(func_name: str, func_args: dict):
    """调用 MCP Server 上的工具"""
    async with streamablehttp_client("http://localhost:8001/mcp") as (
            read_stream,
            write_stream,
            _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(func_name, func_args)
            return parse_tool_result(result)


def call_mcp_tool_sync(func_name: str, func_args: dict):
    return asyncio.run(call_mcp_tool(func_name, func_args))


# =========================
# 主对话逻辑
# =========================
def function_call_playground(
        prompt: str,
        tools=None,
        session_id: str = None,
        system_prompt: str = "You are a helpful AI assistant.",
) -> str:
    session_id = session_id or str(int(time.time()))
    messages = load_messages(session_id)
    if not any(msg["role"] == "system" for msg in messages):
        messages.insert(0, {"role": "system", "content": system_prompt})

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

        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content})
            save_messages(session_id, messages)
            return msg.content

        messages.append(msg.to_dict())

        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            logger.info(f"👉 模型请求调用函数: {func_name}({func_args})")

            if func_name in func_registry:
                try:
                    result = func_registry[func_name](**func_args)
                except Exception as e:
                    result = f"❌ Error: {e}"
            else:
                # 调用 MCP 工具
                try:
                    result = call_mcp_tool_sync(func_name, func_args)
                except Exception as e:
                    result = f"❌ MCP Error: {e}"

            logger.info(f"🔧 执行结果: {result}")

            tool_msg = {
                "role": "tool",
                "content": str(result),
                "tool_call_id": tool_call.id,
            }
            messages.append(tool_msg)

        save_messages(session_id, messages)


# =========================
# 测试
# =========================
if __name__ == "__main__":
    all_tools = my_tools
    try:
        all_tools += get_mcp_tools()
    except Exception as e:
        logger.warning(f"⚠️ 加载 MCP 工具失败: {e}")
    print(function_call_playground("合肥天气怎么样", tools=all_tools))
    # print(function_call_playground("调用远程 add 工具计算 3+5", tools=all_tools, session_id="s1"))
