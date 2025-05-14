from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config import api_key

model = ChatOpenAI(
    api_key=api_key,
    model="Qwen/Qwen2.5-72B-Instruct",
    base_url="https://api.siliconflow.cn/v1",
    temperature=0.01
)
embeddings = OpenAIEmbeddings(model="Pro/BAAI/bge-m3",
                              api_key=api_key,
                              base_url="https://api.siliconflow.cn/v1")
