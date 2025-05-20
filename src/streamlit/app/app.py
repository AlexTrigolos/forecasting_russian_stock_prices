import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import boto3
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from botocore.exceptions import ClientError
from dotenv import load_dotenv

sns.set_style('whitegrid')

load_dotenv()

BUCKET = 'russian-stocks-quotes'

access_key = os.getenv('AWS_ACCESS_KEY_ID')
secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
endpoint_url = 'https://storage.yandexcloud.net'

# Создание клиента S3
s3_client = boto3.client('s3',
                         region_name='ru-central1',
                         aws_access_key_id=access_key,
                         aws_secret_access_key=secret_key,
                         endpoint_url=endpoint_url)

verify = False if os.getenv("VERIFY") == 'False' else True

# response = requests.get(f'{os.getenv("HOST")}/', verify=verify)
# if response.status_code == 200:
#     st.write(response.json())
# else:
#     st.error("Ошибка при получении данных")

# response = requests.get(f'{os.getenv("HOST")}/items/', verify=verify)
# if response.status_code == 200:
#     st.write(response.json())
# else:
#     st.error("Ошибка при получении данных")

# # Заголовок приложения
# st.title("Мое первое приложение на Streamlit")

# # Создание случайных данных
# data = pd.DataFrame(
#     np.random.randn(10, 2),
#     columns=['x', 'y']
# )

# # Отображение данных в таблице
# st.write("Вот случайные данные:")
# st.dataframe(data)

# # Создание графика
# st.line_chart(data)

# Настройка подключения к FastAPI серверу
FAST_API_URL = os.getenv("HOST")

def fetch_secids():
    response = requests.get(f'{FAST_API_URL}/secids/', verify=verify)
    secids = response.json()
    return secids

def fetch_models():
    response = requests.get(f'{FAST_API_URL}/models/', verify=verify)
    models = response.json()
    return models

def display_prediction(model, secid):
    st.title("Предсказательные данные")
    st.write(f"Выбранная модель: {model}, Ценная бумага: {secid}")
    try:
        resp = requests.get(f'{FAST_API_URL}/predict/{model}/{secid}/', verify=verify)
        result = resp.json()
        st.dataframe(result['data_frame'], use_container_width=True)
        for category, images in result['grouped_images'].items():
            st.subheader(category)
            index = 0
            cols = st.columns(2)
            images_count = len(images)
            for name, image in images.items():
                try:
                    s3_client.download_file(BUCKET, image, '/tmp/image')
                    if index % 2 == 0 and index + 1 == images_count:
                        if os.path.exists('/tmp/image'):
                            if name != '':
                                st.subheader(name)
                            st.image('/tmp/image', use_container_width=True)
                    else:
                        with cols[index % 2]:
                            if os.path.exists('/tmp/image'):
                                if name != '':
                                    st.subheader(name)
                                st.image('/tmp/image', use_container_width=True)
                                index += 1
                except ClientError as e:
                    images_count -= 1
    except Exception as e:
        st.error(f"Ошибка: {e}")


def display_mean(model, duration):
    st.title("Предсказательные средние данные")
    st.write(f"Выбранная модель: {model}, с продолжительностью {duration}")
    try:
        resp = requests.get(f'{FAST_API_URL}/predict_mean/{model}/{duration}', verify=verify)
        result = resp.json()
        st.dataframe(result['mean_data_frame'], use_container_width=True)
        for category, images in result['grouped_mean_images'].items():
            st.subheader(category)
            index = 0
            cols = st.columns(2)
            images_count = len(images)
            for name, image in images.items():
                try:
                    s3_client.download_file(BUCKET, image, '/tmp/image')
                    if index % 2 == 0 and index + 1 == images_count:
                        if os.path.exists('/tmp/image'):
                            if name != '':
                                st.subheader(name)
                            st.image('/tmp/image', use_container_width=True)
                    else:
                        with cols[index % 2]:
                            if os.path.exists('/tmp/image'):
                                if name != '':
                                    st.subheader(name)
                                st.image('/tmp/image', use_container_width=True)
                                index += 1
                except ClientError as e:
                    images_count -= 1
    except Exception as e:
        st.error(f"Ошибка: {e}")


st.sidebar.header("Выбор данных")
secids = fetch_secids()
models = fetch_models()

action = st.sidebar.radio("Прогноз по:", ("Акция", "Среднее"))

if action == "Акция":
    selected_model = st.sidebar.selectbox("Модель:", options=models)
    selected_secid = st.sidebar.selectbox("Акция:", options=secids)
    display_prediction(selected_model, selected_secid)
else:
    durations = ['five_years', 'all']
    selected_model = st.sidebar.selectbox("Модель:", options=models)
    selected_duration = st.sidebar.selectbox("Продолжительность:", options=durations)
    display_mean(selected_model, selected_duration)
