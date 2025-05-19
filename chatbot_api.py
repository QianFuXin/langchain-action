import time
from typing import TypedDict, Annotated, Optional

from flask import Flask, request, Response, jsonify
from langchain_core.messages import HumanMessage, AIMessage, trim_messages, SystemMessage, AnyMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, add_messages
from utils import model


class MyMessagesState(TypedDict):
    # 原始消息列表（支持多轮对话）
    messages: Annotated[list[AnyMessage], add_messages]
    # 可选的 system prompt，用于注入 SystemMessage
    system_prompt: Optional[str]


app = Flask(__name__)
# 定义 LangGraph 工作流
workflow = StateGraph(state_schema=MyMessagesState)


# 调用模型的方法
def call_model(state: MyMessagesState):
    # 获取消息列表
    messages = state["messages"]

    # 提取 system_prompt（如果有）
    system_prompt = state.get("system_prompt")
    if system_prompt:
        system_message = SystemMessage(system_prompt)
        # 检查第一条是否是 SystemMessage
        if messages and isinstance(messages[0], SystemMessage):
            # messages[0] = system_message  # 替换
            pass
        else:
            messages = [system_message] + messages  # 插入

    trimmed_messages = trimmer.invoke(messages)
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


def generate(thread_id, message, system_prompt=None):
    input_messages = [HumanMessage(message)]
    config = {
        "configurable": {"thread_id": thread_id}
    }
    inputs = {
        "messages": input_messages,
    }
    if system_prompt:
        inputs["system_prompt"] = system_prompt

    data = compiled_app.invoke(inputs, config)["messages"]
    return data[-1].content


def generate_stream(thread_id, message, system_prompt=None):
    input_messages = [HumanMessage(message)]
    config = {
        "configurable": {"thread_id": thread_id}
    }
    inputs = {
        "messages": input_messages,
    }
    if system_prompt:
        inputs["system_prompt"] = system_prompt

    for chunk, metadata in compiled_app.stream(inputs, config, stream_mode="messages"):
        if isinstance(chunk, AIMessage):
            yield chunk.content


@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    data = request.json
    thread_id = data.get("thread_id", str(time.time()))
    message = data.get("message")
    system_prompt = data.get("system_prompt")

    if not message:
        return jsonify({"error": "message is required"}), 400

    return Response(generate_stream(thread_id, message, system_prompt), content_type="text/plain; charset=utf-8")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    thread_id = data.get("thread_id", str(time.time()))
    message = data.get("message")
    system_prompt = data.get("system_prompt")

    if not message:
        return jsonify({"error": "message is required"}), 400

    return Response(generate(thread_id, message, system_prompt), content_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
    """
    curl -N -X POST "http://127.0.0.1:8000/chat_stream" \
     -H "Content-Type: application/json" \
     -d '{"thread_id": "abc1234", "message": "随机生成100字作文","system_prompt":"用海盗的口吻跟我对话"}'
     
    curl -N -X POST "http://127.0.0.1:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"thread_id": "abc123", "message": "你好","system_prompt":"用海盗的口吻跟我对话"}'
     
     
    """
