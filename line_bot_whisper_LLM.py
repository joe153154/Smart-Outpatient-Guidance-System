import os
import re
import csv
import sys
import time
import json
import torch
import pyttsx3
import warnings
import keyboard
import warnings
import numpy as np
import pandas as pd
import transformers
import torch.nn as nn
import seaborn as sns
import matplotlib as mpl
import speech_recognition as sr
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List
from pylab import rcParams
from pydub import AudioSegment
from translate import Translator
from datasets import load_dataset
from torch.utils.data import DataLoader
from flask import Flask, request, abort
from peft import AutoPeftModelForCausalLM
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, AudioMessage, TextSendMessage
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig, WhisperProcessor, WhisperForConditionalGeneration
import librosa
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


def model_processing(model, tokenizer, input_text):
    prompt = "你是我的一名助理，我會問你一些問題，你需要透過使用繁體中文回答問題，然後給我用一句話說完。現在你是我的朋友，請你根據話題和我聊天。" + "\n" + f"{input_text}"
    response, history = model.chat(tokenizer, prompt, history=None)
    return response

app = Flask(__name__)
line_bot_api = LineBotApi('your linbot api key')
handler = WebhookHandler('your Webhook api key')

device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else "cpu")
print(f"Using the GPU {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "Using CPU")

text_model_path = "text_model"
audio_model_path = "audio_model_mix"
text_tokenizer = AutoTokenizer.from_pretrained(text_model_path, trust_remote_code=True)
text_model = AutoModelForCausalLM.from_pretrained(text_model_path, device_map={"": 0}, trust_remote_code=True)
audio_processor = WhisperProcessor.from_pretrained(audio_model_path, language="chinese", task="transcribe")
audio_model = WhisperForConditionalGeneration.from_pretrained(audio_model_path)


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
    UserID = event.source.user_id
    audio_data_path = f"./static/{UserID}.wav"
    audio_content = line_bot_api.get_message_content(event.message.id)
    with open(audio_data_path, 'wb') as fd:
        for chunk in audio_content.iter_content():
            fd.write(chunk)
    fd.close()
    input_text = audio_process(audio_data_path, audio_model, audio_processor, device)
    print(f"audio input: {input_text}")
    results = model_processing(text_model, text_tokenizer, input_text)
    print(f"language model Response:{results}")

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=results))


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)