import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import QuerySQLDatabaseTool
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict, Annotated
from utils import db, model


# ========== 定义数据结构 ==========
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str


# ========== 生成 SQL 查询语句 ==========
SQL_TEMPLATE = """
你是一个擅长数据库操作的专家。你的任务是根据用户的问题生成语法正确的 {dialect} 查询语句。

要求如下：
1. 查询语句必须符合 SQL 语法，并能在实际数据库中执行。
2. 如果用户没有说明需要返回多少条数据，默认使用 LIMIT {top_k} 限制返回结果数量。
3. 不要使用 SELECT *，只选择与问题相关的字段。
4. 只能使用以下表结构和字段：
{table_info}
5. 查询语句必须使用 markdown 三引号代码块格式包裹，并指定 SQL 语言类型，格式如下：
```sql
SELECT column1, column2 FROM your_table WHERE condition LIMIT 10;
```
下面是用户的问题：

{input}

请生成符合要求的 SQL 查询语句。
"""

query_prompt_template = ChatPromptTemplate.from_template(SQL_TEMPLATE)


def extract_sql_from_code_block(text: str) -> str:
    """提取 markdown 格式 SQL 代码块中的 SQL 内容"""
    match = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def write_query(state: State):
    """根据用户问题生成 SQL 查询语句"""
    prompt = query_prompt_template.invoke({
        "dialect": db.dialect,
        "top_k": 10,
        "table_info": db.get_table_info(),
        "input": state["question"],
    })
    query = extract_sql_from_code_block(model.invoke(prompt).content)
    return {"query": query}


# ========== 安全执行 SQL ==========
def execute_query(state: State):
    """执行 SQL 查询语句"""
    try:
        query_tool = QuerySQLDatabaseTool(db=db)
        result = query_tool.invoke(state["query"])
    except Exception as e:
        result = f"SQL 执行错误: {e}"
    return {"result": result}


# ========== 生成最终回答 ==========
def generate_answer(state: State):
    """将 SQL 查询结果作为上下文进行回答"""
    prompt = (
        "根据以下用户问题、对应的 SQL 查询语句和 SQL 查询结果，回答用户问题：\n\n"
        f"问题: {state['question']}\n"
        f"SQL 查询: {state['query']}\n"
        f"SQL 结果: {state['result']}"
    )
    response = model.invoke(prompt)
    return {"answer": response.content}


# ========== 构建执行图 ==========
graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query")

# 添加中断点以实现交互审批
graph = graph_builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["execute_query"]
)

# ========== 执行流程 ==========
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    while 1:
        question = input("用户：")
        # 初步执行直到中断点
        for step in graph.stream({"question": question}, config, stream_mode="updates"):
            if "write_query" in step:
                print("即将执行" + step["write_query"]["query"])
        # 用户确认是否执行 SQL
        try:
            user_approval = input("是否继续执行 SQL 查询？(yes/no): ")
        except Exception:
            user_approval = "no"

        if user_approval.strip().lower() == "yes":
            for step in graph.stream(None, config, stream_mode="updates"):
                if "execute_query" in step:
                    print("执行结果" + step["execute_query"]["result"])
                if "generate_answer" in step:
                    print("最终答案:" + step["generate_answer"]["answer"])

        else:
            print("操作已取消。")
