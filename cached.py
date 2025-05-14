"""
缓存
"""
import logging

from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from utils import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

set_llm_cache(InMemoryCache())

print(model.invoke("跟我说一个笑话").content)
print("-" * 20)
print(model.invoke("跟我说一个笑话").content)
print("-" * 20)
print(model.invoke("跟我说一个关于猫的笑话").content)
