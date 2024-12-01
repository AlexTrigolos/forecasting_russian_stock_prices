import streamlit as st
import pandas as pd
import numpy as np
import requests

response = requests.get("http://fastapi:8000/")
if response.status_code == 200:
    st.write(response.json())
else:
    st.error("Ошибка при получении данных")

response = requests.get("http://fastapi:8000/items/")
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
