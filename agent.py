"""
智能体
"""
import time
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from utils import *
from tools import *

memory = MemorySaver()
tools = [execute_python_code, read_file]
agent_executor = create_react_agent(model, tools, checkpointer=memory)

config = {"configurable": {"thread_id": str(int(time.time()))}}

while 1:
    question = input("用户：")
    print("机器人：", end="")
    for chunk, metadata in agent_executor.stream({"messages": [HumanMessage(question)]},
                                                 config,
                                                 stream_mode="messages"):
        if isinstance(chunk, ToolMessage):
            print("=" * 20 + "tool" + "=" * 20)
            print(chunk)
            print("=" * 20 + "tool" + "=" * 20)
        if isinstance(chunk, AIMessage):
            print(chunk.content, end="")
    print()
