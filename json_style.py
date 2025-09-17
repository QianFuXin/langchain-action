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

from pydantic import BaseModel, Field


class User(BaseModel):
    """
    User model representing a user in the system.
    """
    id: int = Field(
        ...,
        description="Unique identifier of the user",
        ge=1,
        example=123
    )
    name: str = Field(
        ...,
        description="Full name of the user",
        min_length=1,
        max_length=50,
        example="Alice"
    )
    is_active: bool = Field(
        default=True,
        description="Indicates whether the user account is active",
        example=True
    )


text = "昨天注册了一个新用户 Alice，她的ID是123，账号目前是激活状态。"


def schema_of(model_cls):
    return model_cls.model_json_schema() if hasattr(model_cls, "model_json_schema") else model_cls.schema()


prompt = f"""你是一个信息抽取助手。你的任务是从给定文本中抽取实体，并将结果严格按照指定的 JSON Schema 格式输出。

# 抽取规则
- 必须严格遵守 JSON Schema 中定义的字段、类型和约束。
- 如果文本中没有提到某个字段，请使用 `null` 或合理的默认值。
- 输出必须是合法的 JSON，不要添加额外的解释。

# JSON Schema
{schema_of(User)}

# 待抽取文本
{text}

# 输出要求
请返回一个 JSON 对象，完全符合上述 JSON Schema。"""

print(model.invoke(prompt).content)
