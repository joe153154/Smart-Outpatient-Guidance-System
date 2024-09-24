import os
import re
import csv
import sys
import time
import json
import torch
import openai
import librosa
import pyttsx3
import warnings
import numpy as np
import pandas as pd
import transformers
import torch.nn as nn
import seaborn as sns
import matplotlib as mpl
import torch.nn.functional as F
import matplotlib.pyplot as plt
from time import time
from typing import List
from pylab import rcParams
from pydub import AudioSegment
from datasets import load_dataset
from torch.utils.data import DataLoader
from flask import Flask, request, abort
from peft import AutoPeftModelForCausalLM
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, AudioMessage, TextSendMessage
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig, WhisperProcessor, WhisperForConditionalGeneration

warnings.simplefilter("ignore")


def audio_process(audio_data_path, model, processor, device):
    model = model.to(device)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")
    audio_data, sample_rate = librosa.load(audio_data_path, sr=16000)
    input_features = processor(audio_data, return_tensors="pt", sampling_rate=sample_rate).input_features
    input_features = input_features.to(device)
    predicted_ids = model.generate(input_features, max_length=500)
    result = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return result[0]

device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else "cpu")
print(f"Using the GPU {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "Using CPU")

app = Flask(__name__)

line_bot_api = LineBotApi('your linbot api key')
handler = WebhookHandler('your Webhook api key')

openai.api_key = "your openAI api key"
users = {}

a = 0
messages = []
audio_model_path = "audio_model_mix"
audio_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3", language="zh", task="transcribe")
audio_model = WhisperForConditionalGeneration.from_pretrained(audio_model_path)
relus = [
    f'<規則1> 現在你是一名醫療人員，請你在根據我的敘述判斷我的情況',
    f'<規則2> 當你問完3句話之後如果你知道病患的狀況後，你需要告知病患該選擇哪個門診與有可能的病名',
    f'<規則3> 請使用輸入的文字可能會有些字會出現同音但不同字，請你自行修正文在進行推理',
    f'<規則4> 在回覆的時候只能夠使用一個問句',
    f'<規則5> 你不能回答醫療以外的所有問題',
    f'<規則6> 你每次的回覆不能超過10個字',
    f'<規則7> 接下來的回答都需要用zh-tw回答',
    f'<規則8> 你需要記住我的狀況',
    f'<規則9> 然後不要問同樣的問題',
    f'<規則10> 請依照先詢問持續時間，再問具體是那裡不舒服，然後問有沒有其他症狀，最後在跟使病患說該前往哪個門診就醫',
    f'<規則11> 請務必都要遵守以上規則',
]

@app.route("/", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(250)
    return 'OK'


# 處理訊息
@handler.add(MessageEvent, message=AudioMessage)
def handle_audio_message(event):
    ID = event.source.user_id
    # 判斷用戶是否初次進入系統
    if ID not in users:
        users[ID] = {'Dialog': [{}], 'Time': time()}

    dialog_time = time() - users[ID]['Time']
    if  dialog_time > 90:
        users[ID] = {'Dialog':[{}], 'Time':time()}

    audio_data_path = f"./static/static.wav"

    audio_content = line_bot_api.get_message_content(event.message.id)
    with open(audio_data_path, 'wb') as fd:
        for chunk in audio_content.iter_content():
            fd.write(chunk)
    fd.close()
    input_text = audio_process(audio_data_path, audio_model, audio_processor, device)
    input_text = input_text.replace(" ", "")
    print(f"audio input: {input_text}")

    # 新增用戶對話
    messages.append({"role": "system", "content": ''.join(relus)+'\n'+f'{input_text}'})

    # ChatGPT回復設定
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
    )

    result = ''
    for choice in response.choices:
        result += choice.message.content
    print(f'GPT Response:{result}')
    messages.append({"role": "assistant", "content": result})

    # Line 回復
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=result))

    # 更新時間
    users[ID]['Time'] = time()


if __name__ == "__main__":
    app.run(port=5000,debug=True)