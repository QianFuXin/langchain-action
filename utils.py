from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config import api_key, mysql_url

model = ChatOpenAI(
    api_key=api_key,
    model="Qwen/Qwen2.5-72B-Instruct",
    base_url="https://api.siliconflow.cn/v1",
    temperature=0.01
)
embeddings = OpenAIEmbeddings(model="Pro/BAAI/bge-m3",
                              api_key=api_key,
                              base_url="https://api.siliconflow.cn/v1")
db = SQLDatabase.from_uri(mysql_url)


def asr(filename):
    import requests
    from config import api_key
    url = "https://api.siliconflow.cn/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    data = {"model": "FunAudioLLM/SenseVoiceSmall"}
    files = {"file": open(filename, "rb")}

    response = requests.post(url, headers=headers, data=data, files=files)
    return response.json()["text"]


def tts(word, output_filename="output.mp3", play=False):
    from utils import api_key
    import requests
    from playsound import playsound
    url = "https://api.siliconflow.cn/v1/audio/speech"

    payload = {
        "model": "FunAudioLLM/CosyVoice2-0.5B",
        "input": f"Can you say it with a happy emotion? <|endofprompt|>{word}",
        "voice": "FunAudioLLM/CosyVoice2-0.5B:alex",
        "response_format": "mp3",
        "sample_rate": 32000,
        "stream": True,
        "speed": 1,
        "gain": 0
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    # 保存二进制音频内容
    if response.status_code == 200:
        with open(output_filename, "wb") as f:
            f.write(response.content)
        print(f"音频已保存为：{output_filename}")
        if play:
            playsound(output_filename)
    else:
        print(f"请求失败，状态码：{response.status_code}")
        print(response.text)
