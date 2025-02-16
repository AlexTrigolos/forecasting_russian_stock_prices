import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from dotenv import load_dotenv

load_dotenv()

verify = False if os.getenv("VERIFY") == 'False' else True

response = requests.get(f'{os.getenv("HOST")}/', verify=verify)
if response.status_code == 200:
    st.write(response.json())
else:
    st.error("Ошибка при получении данных")

response = requests.get(f'{os.getenv("HOST")}/items/', verify=verify)
if response.status_code == 200:
    st.write(response.json())
else:
    st.error("Ошибка при получении данных")

# Заголовок приложения
st.title("Мое первое приложение на Streamlit")

# Создание случайных данных
data = pd.DataFrame(
    np.random.randn(10, 2),
    columns=['x', 'y']
)

# Отображение данных в таблице
st.write("Вот случайные данные:")
st.dataframe(data)

# Создание графика
st.line_chart(data)
