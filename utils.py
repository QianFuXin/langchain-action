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


def text2image(prompt):
    from utils import api_key
    import requests
    from urllib.parse import unquote, urlparse
    import os
    url = "https://api.siliconflow.cn/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "Kwai-Kolors/Kolors",
        "prompt": prompt,
        "image_size": "1024x1024",
        "batch_size": 1,
        "num_inference_steps": 20,
        "guidance_scale": 7.5
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        # 3. 提取图像 URL
        image_url = result["images"][0]["url"]
        print("✅ Image URL:", image_url)

        # 4. 下载图片内容
        image_response = requests.get(image_url)
        if image_response.status_code == 200:
            # 5. 解析文件名（可自定义）
            parsed_url = urlparse(unquote(image_url))
            filename = os.path.basename(parsed_url.path)
            # 或自定义：filename = f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            # 6. 保存到本地
            with open(filename, "wb") as f:
                f.write(image_response.content)
            print(f"📁 Image saved as {filename}")
        else:
            print(f"❌ Failed to download image: {image_response.status_code}")
    else:
        print(f"❌ API call failed: {response.status_code}")
        print(response.text)
