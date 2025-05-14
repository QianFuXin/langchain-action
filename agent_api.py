import time
from flask import Flask, request, Response, jsonify
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from utils import *
from tools import *

app = Flask(__name__)

# 构建 Agent + 工具 + 内存
memory = MemorySaver()
tools = [execute_python_code, read_file]
agent_executor = create_react_agent(model, tools, checkpointer=memory)


# ===== 流式处理生成器 =====
def generate_stream(thread_id, message):
    input_messages = [HumanMessage(message)]
    config = {"configurable": {"thread_id": thread_id}}

    for chunk, metadata in agent_executor.stream(
            {"messages": input_messages}, config, stream_mode="messages"
    ):
        if isinstance(chunk, AIMessage):
            yield chunk.content


# ===== 接口：流式 =====
@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    data = request.json
    thread_id = data.get("thread_id", str(time.time()))
    message = data.get("message")
    if not message:
        return jsonify({"error": "message is required"}), 400

    return Response(generate_stream(thread_id, message), content_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)

"""
测试方法：

curl -N -X POST "http://127.0.0.1:8001/chat_stream" \
 -H "Content-Type: application/json" \
 -d '{"thread_id": "abc123", "message": "hello_word.py的运行结果是什么"}'

"""
