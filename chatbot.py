"""
聊天机器人
"""
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from utils import *

# 定义一个图
workflow = StateGraph(state_schema=MessagesState)


# 定义调用模型的方法
def call_model(state: MessagesState):
    trimmed_messages = trimmer.invoke(state["messages"])
    response = model.invoke(trimmed_messages)
    return {"messages": response}


trimmer = trim_messages(
    strategy="last",
    token_counter=len,
    # 最多保留20个对话（包括系统、用户、AI三部分,如果没有系统消息，则最多保留十轮对话）
    max_tokens=20,
    include_system=True,
    allow_partial=False,
    start_on="human",
)
# 定义图中的一个节点
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# 添加内存存储
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
# 配置
config = {"configurable": {"thread_id": "abc123"}}

while 1:
    question = input("用户：")
    input_messages = [HumanMessage(question)]
    print("机器人：", end="")
    for chunk, metadata in app.stream({"messages": input_messages}, config, stream_mode="messages"):
        if isinstance(chunk, AIMessage):
            print(chunk.content, end="")
    print()
