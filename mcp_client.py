import asyncio
import time

from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.tools import load_mcp_tools

from utils import model
config = {"configurable": {"thread_id": str(int(time.time()))}}
memory = MemorySaver()


async def chat(agent):
    print("💬 请输入问题，输入 'exit' 退出。\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit"]:
            print("👋 再见！")
            break

        try:
            async for chunk, metadata in agent.astream(
                    {"messages": [HumanMessage(question)]},
                    config,
                    stream_mode="messages"
            ):
                if isinstance(chunk, ToolMessage):
                    print("=" * 20 + "tool" + "=" * 20)
                    print(chunk)
                    print("=" * 20 + "tool" + "=" * 20)
                if isinstance(chunk, AIMessage):
                    print(chunk.content, end="")
            print()
        except Exception as e:
            print(f"⚠️ 调用出错: {e}\n")


async def main():
    async with streamablehttp_client("http://localhost:8001/mcp/") as (read, write, _):
        async with ClientSession(read, write) as session:
            # 初始化连接
            await session.initialize()

            # 获取 MCP 工具
            tools = await load_mcp_tools(session)
            print(f"✅ 已加载工具 {len(tools)} 个")

            # 创建 agent（带记忆）
            agent = create_react_agent(model, tools, checkpointer=memory)

            # 进入交互式循环
            await chat(agent)


if __name__ == "__main__":
    asyncio.run(main())