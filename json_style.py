"""
字段抽取
"""
from utils import *

from typing_extensions import Annotated, TypedDict


# 定义一个笑话的数据结构
class Joke(TypedDict):
    """
    用于向用户讲述的笑话。
    """

    setup: Annotated[str, ..., "笑话的开场白"]

    punchline: Annotated[str, ..., "笑话的包袱"]
    rating: Annotated[int, ..., "笑话的趣味评分，范围从 1 到 10"]


# 使用 LLM 模型，并将其输出结构化为 Joke 类型
structured_llm = model.with_structured_output(Joke)

# 调用 LLM，生成一个关于猫的笑话
res = structured_llm.invoke("说一个关于认的冷笑话")

# 打印生成的笑话
print(res)
