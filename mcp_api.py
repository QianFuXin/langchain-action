import time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from typing import AsyncGenerator

from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain_mcp_adapters.tools import load_mcp_tools

from langgraph.prebuilt import create_react_agent
from utils import model

app = FastAPI()

# 全局内存 & 配置
memory = MemorySaver()
config = {"configurable": {"thread_id": str(int(time.time()))}}


@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    question = body.get("question")
    if not question:
        return JSONResponse({"error": "缺少参数 question"}, status_code=400)

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            async with streamablehttp_client("http://localhost:8001/mcp") as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await load_mcp_tools(session)
                    agent = create_react_agent(model, tools, checkpointer=memory)

                    async for chunk, metadata in agent.astream(
                            {"messages": [HumanMessage(question)]},
                            config,
                            stream_mode="messages"
                    ):
                        if isinstance(chunk, ToolMessage):
                            yield f"\n[TOOL]\n{chunk.json()}\n[/TOOL]\n"
                        elif isinstance(chunk, AIMessage):
                            yield chunk.content
        except Exception as e:
            yield f"\n⚠️ 调用出错: {e}\n"

    return StreamingResponse(event_stream(), media_type="text/plain")


# uvicorn mcp_api:app --host 0.0.0.0 --port 5001 --reload
"""
curl -N -X POST http://localhost:5001/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "你可以使用哪些工具"}'
"""
