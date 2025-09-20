"""
支持本地工具 + 多个远程 MCP 工具调用（覆盖模式）
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
        return result.structuredContent.get("result")
    if hasattr(result, "content") and result.content:
        return getattr(result.content[0], "text", None)
    return None


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
# 全局工具注册表（覆盖模式）
# =========================
TOOL_REGISTRY = {}  # { func_name: {"type": "local"/"mcp", "handler": ...} }
all_tools = []


def local_tool(description=""):
    """装饰器：注册本地工具并自动加入 all_tools"""

    def wrapper(func):
        schema = register_local_tool(func, description)
        all_tools.append(schema)
        return func

    return wrapper


def register_local_tool(func, description=""):
    """注册本地工具（覆盖模式）"""
    sig = inspect.signature(func)
    hints = typing.get_type_hints(func)

    TYPE_MAP = {int: "number", float: "number", str: "string", bool: "boolean"}

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

    TOOL_REGISTRY[func.__name__] = {"type": "local", "handler": func}
    return schema


def register_mcp_tool(server_url: str, tool):
    """注册 MCP 工具（覆盖模式，不加前缀）"""
    func_name = tool.name

    schema = {
        "type": "function",
        "function": {
            "name": func_name,
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

    TOOL_REGISTRY[func_name] = {"type": "mcp", "handler": (server_url, tool.name)}
    return schema


def execute_tool(func_name: str, func_args: dict):
    """统一执行工具（覆盖模式）"""
    if func_name not in TOOL_REGISTRY:
        return f"❌ Unknown tool: {func_name}"

    entry = TOOL_REGISTRY[func_name]

    if entry["type"] == "local":
        func = entry["handler"]
        try:
            return func(**func_args)
        except Exception as e:
            return f"❌ Local Error: {e}"

    elif entry["type"] == "mcp":
        server_url, real_name = entry["handler"]
        try:
            return call_mcp_tool_sync(server_url, real_name, func_args)
        except Exception as e:
            return f"❌ MCP Error: {e}"


# =========================
# MCP 工具支持
# =========================
async def fetch_mcp_tools(server_url: str):
    async with streamablehttp_client(server_url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()
            tools = result.tools if hasattr(result, "tools") else result[-1]
            return tools


def get_all_mcp_tools(servers):
    """加载所有 MCP server 的工具并注册"""
    all_tools = []
    for server in servers:
        try:
            tools = asyncio.run(fetch_mcp_tools(server))
            for t in tools:
                all_tools.append(register_mcp_tool(server, t))
        except Exception as e:
            logger.warning(f"⚠️ 加载 MCP 工具失败: {server}, {e}")
    return all_tools


async def call_mcp_tool(server_url: str, func_name: str, func_args: dict):
    async with streamablehttp_client(server_url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(func_name, func_args)
            return parse_tool_result(result)


def call_mcp_tool_sync(server_url: str, func_name: str, func_args: dict):
    return asyncio.run(call_mcp_tool(server_url, func_name, func_args))


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

            result = execute_tool(func_name, func_args)

            logger.info(f"🔧 执行结果: {result}")

            tool_msg = {
                "role": "tool",
                "content": str(result),
                "tool_call_id": tool_call.id,
            }
            messages.append(tool_msg)

        save_messages(session_id, messages)


@local_tool("Calculate the product of two numbers")
def mul(a: float, b: float) -> float:
    return a * b


@local_tool("Compare two numbers, which one is bigger")
def compare(a: float, b: float) -> str:
    if a > b:
        return f"{a} is greater than {b}"
    elif a < b:
        return f"{b} is greater than {a}"
    return f"{a} is equal to {b}"


if __name__ == "__main__":
    # MCP server 列表
    MCP_SERVERS = [
        "http://localhost:8001/mcp",
        "http://localhost:8002/mcp",
    ]

    # 远程 MCP 工具
    all_tools += get_all_mcp_tools(MCP_SERVERS)
    # 测试调用
    print(function_call_playground("计算12*1111", tools=all_tools))
