import time

from flask import Flask, request, Response, jsonify
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from utils import *

app = Flask(__name__)
# 定义 LangGraph 工作流
workflow = StateGraph(state_schema=MessagesState)


# 调用模型的方法
def call_model(state: MessagesState):
    trimmed_messages = trimmer.invoke(state["messages"])
    response = model.invoke(trimmed_messages)
    return {"messages": response}


# 配置消息修剪策略
trimmer = trim_messages(
    strategy="last",
    token_counter=len,
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
compiled_app = workflow.compile(checkpointer=memory)


def generate_stream(thread_id, message):
    input_messages = [HumanMessage(message)]
    config = {"configurable": {"thread_id": thread_id}}
    for chunk, metadata in compiled_app.stream({"messages": input_messages}, config, stream_mode="messages"):
        if isinstance(chunk, AIMessage):
            yield chunk.content + "\n"


@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    data = request.json
    thread_id = data.get("thread_id", str(time.time()))
    message = data.get("message")
    if not message:
        return jsonify({"error": "message is required"}), 400

    return Response(generate_stream(thread_id, message), content_type="text/plain; charset=utf-8")


def generate(thread_id, message):
    input_messages = [HumanMessage(message)]
    config = {"configurable": {"thread_id": thread_id}}
    data = compiled_app.invoke({"messages": input_messages}, config)["messages"]
    return data[-1].content


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    thread_id = data.get("thread_id", str(time.time()))
    message = data.get("message")
    if not message:
        return jsonify({"error": "message is required"}), 400
    return Response(generate(thread_id, message), content_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
    """
    curl -N -X POST "http://127.0.0.1:8000/chat_stream" \
     -H "Content-Type: application/json" \
     -d '{"thread_id": "abc123", "message": "随机生成100字作文"}'
     
    curl -N -X POST "http://127.0.0.1:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"thread_id": "abc123", "message": "我说的上句话是什么"}'
     
    """
