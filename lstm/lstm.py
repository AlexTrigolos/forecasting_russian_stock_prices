import copy
import os
import boto3
import traceback
import io
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from botocore.exceptions import ClientError, NoCredentialsError
from traceback import format_exc
from sklearn.metrics import mean_squared_error

os.environ['AWS_ACCESS_KEY_ID'] = <access_key>
os.environ['AWS_SECRET_ACCESS_KEY'] = <secret_access_key>

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

def download_object_from_s3(key):
    response = s3_client.get_object(Bucket=BUCKET, Key=key)
    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        print(f"Успешно получен из {BUCKET}/{key}")
    else:
        print(f"Ошибка при получении: {response['ResponseMetadata']['HTTPStatusCode']}")
    return response['Body'].read()

def download_info_from_s3(dir, secid):
    key = f'{dir}secids/{secid}/{secid}_info.pkl'
    response = download_object_from_s3(key)
    data = json.loads(response)
    data['miss_index'] = pd.Index(np.array(data['miss_index']))
    return data

def download_data_frame_from_s3(dir, secid):
    key = f'{dir}secids/{secid}/{secid}_data_frame.pkl'
    response = download_object_from_s3(key)
    buffer = io.BytesIO(response)
    data = pd.read_pickle(buffer)
    data['TRADEDATE'] = pd.to_datetime(data['TRADEDATE'])
    return data

def download_secid_names(dir):
    key = f'{dir}secid_names.pkl'
    return json.loads(download_object_from_s3(key))

def fit_secids_from_s3(dir, secids=None):
    data = {}
    try:
        directories = download_secid_names(dir)
        for secid in directories:
            if (secids is None or secid in secids) and secid >= 'RBCM':
                if secids is not None:
                    secids.remove(secid)
                fit_secid(secid, download_data_frame_from_s3(dir, secid))
    except Exception as e:
        error_message = f"Неизвестная ошибка: {str(e)}"
        error_context = traceback.format_exc()
        print(f"{error_message}\nКонтекст ошибки:\n{error_context}")
    if secids is not None and len(secids) > 0:
        print(f'Не нашли {secids}')
    return data

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred, epsilon=1e-6):
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def metrics(y_true, y_pred):
  # корень из квадратичной ошибки, возвращает ошибку в тех же единицах, что и целевая переменная
  rmse_score = rmse(y_true, y_pred)
  # измеряет ошибку в процентах и позволяет легко интерпретировать результаты
  mape_score = mape(y_true, y_pred)
  return tuple([rmse_score, mape_score])

from sklearn.preprocessing import MinMaxScaler
import copy
import datetime
import pickle
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.initializers import glorot_uniform
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.losses import Huber, MeanSquaredError
from keras.layers import LayerNormalization
from keras.optimizers import AdamW
import tensorflow as tf
import random
import math

def divisors(n):
    divisors = list()
    for i in range(1, n):
        if n % i == 0:
            divisors.append(i)
    return divisors

lags = { 1: 'lag_1', 2: 'lag_2', 3: 'lag_3', 4: 'lag_4', 5: 'lag_week', 10: 'lag_2_weeks',
        21: 'lag_month', 62: 'lag_3_months', 124: 'lag_half_year', 247: 'lag_year',
        371: 'lag_year_with_half', 495: 'lag_2_years', 742: 'lag_3_years' }
rev_lags = { 'lag_1': 1, 'lag_2': 2, 'lag_3': 3, 'lag_4': 4, 'lag_week': 5, 'lag_2_weeks': 10,
            'lag_month': 21, 'lag_3_months': 62, 'lag_half_year': 124, 'lag_year': 247,
            'lag_year_with_half': 371, 'lag_2_years': 495, 'lag_3_years': 742 }

def lr_scheduler(epoch, lr):
    if epoch < 50:
        return lr
    else:
        return lr * 0.95

def upload_models_data_to_s3(secid, model_name, body):
    key = f'predictions/{secid}/{model_name}.pkl'
    response = s3_client.put_object(Bucket=BUCKET, Key=key, Body=pickle.dumps(body))
    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        print(f"Успешно сохранен в {BUCKET}/{key}")
    else:
        print(f"Ошибка при сохранении: {response['ResponseMetadata']['HTTPStatusCode']}")

def fit_secid(secid, data_frame):
  random.seed(42)
  # Получаем данные по бумаге и удаляем дату
  secid_data = data_frame[['TRADEDATE', 'CLOSE']]
  subset = pd.to_datetime(secid_data['TRADEDATE'])
  secid_data = secid_data.drop('TRADEDATE', axis=1)

  # добавляем дату по отдельности
  secid_data.loc[:, 'year'] = subset.dt.year
  secid_data.loc[:, 'month'] = subset.dt.month
  secid_data.loc[:, 'day'] = subset.dt.day

  # Добавляем отступы по возможным корреляциям (очень сложно выбрать нормальные отсутпы по причине того, что торги на бирже не нормированы, есть праздники, переносы, блокировки торгов, переезд компаний и другое)
  # но в среднем интернет выдал 247 с хвостиком рабочих дней в году, что я уже пытался нормально разделить, например для месяца получается 21 торговый день, хоть дней примерно 30
  for lag_name, lag_num in rev_lags.items():
    secid_data[lag_name] = secid_data['CLOSE'].shift(lag_num)

  # Далее убираются строки по лагам, которые не имеют данных, если данных остается меньше чем на 3 месяца или меньше 10% от начальных данных, то удаляется полностью колонка
  # Потому что может быть ситуация, что данных на 3 года и 2 месяца, и только 2 месяца будут иметь лаг в 3 года, а я не хочу удалять так много данных
  # Если же данных достаточно, то это будет самый большой лаг и удаляются строки, в которых по этому лагу пропуски. И переназначаем индексы
  # lags = ['lag_3_years', 'lag_2_years', 'lag_year_with_half', 'lag_year', 'lag_half_year', 'lag_3_months', 'lag_month', 'lag_2_weeks', 'lag_week', 'lag_4', 'lag_3', 'lag_2', 'lag_1']
  lag_names = list(reversed(lags.values()))
  for lag in lag_names:
    if secid_data[lag].isnull().sum() > 0:
      temp = secid_data.dropna(subset=[lag])
      if len(temp) < 62 or len(temp) * 10 < len(secid_data):
        secid_data = secid_data.drop(columns=[lag])
      else:
        secid_data = temp
        break
  secid_data = secid_data.reset_index().drop('index', axis=1)

  # Разбиваем данные, в валидацию идет 20%
  train_size = int(len(secid_data) * 0.8)
  train, test = secid_data[:train_size], secid_data[train_size:]

  valid_columns = [column for column in list(reversed(lags.values())) if column in test.columns.tolist()]
  valid_columns.append('CLOSE')
  # Очищаем те данные, что в валидацию попали из валидационных данных (чтобы не пытаться использовать известные реальные целевые значения для обучения)
  for lag_name, lag_num in rev_lags.items():
    if lag_name in valid_columns:
      test.loc[test[lag_name] == test['CLOSE'].shift(lag_num), lag_name] = np.nan

  # нормализуем лаги (не трогаем таргет и даты)
  scaler = MinMaxScaler()
  # valid_columns = ['lag_3_years', 'lag_2_years', 'lag_year_with_half', 'lag_year', 'lag_half_year', 'lag_3_months', 'lag_month', 'lag_2_weeks', 'lag_week', 'lag_4', 'lag_3', 'lag_2', 'lag_1', 'CLOSE']
  train.loc[:, valid_columns] = scaler.fit_transform(train[valid_columns])
  # val[valid_columns] = scaler.transform(val[valid_columns])

  base_models = [
      {
          'name': 'lstm',
          'model': 'LSTM',
          'grid_params': {},
          'importances_name': ''
      }
  ]
  models_data = copy.deepcopy(base_models)
  for data in models_data:
    print(f"Обучение {data['model']} на {secid}")
    train_data = copy.deepcopy(train)
    test_data = copy.deepcopy(test)

    X_train = train_data.drop('CLOSE', axis=1)
    y_train = train_data['CLOSE']

    features = X_train.shape[1]
    best_first_10_mape_sum = float('inf')
    best_data = None
    for timesteps in divisors(X_train.shape[0]):
      metric_scores = list()
      predictions = list()
      print(timesteps)
      X_train_reshaped = X_train.to_numpy().reshape((X_train.shape[0] // timesteps, timesteps, features))
      # обучаем модель
      model = Sequential()
      model.add(LSTM(48, input_shape=(timesteps, features),
                    return_sequences=True,
                    kernel_initializer='he_normal',
                    recurrent_dropout=0.1))
      model.add(LayerNormalization())
      model.add(Dropout(0.2))
      model.add(LSTM(24, return_sequences=False))
      model.add(LayerNormalization())
      model.add(Dense(16, activation='relu'))
      model.add(Dense(1))

      optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
      model.compile(optimizer=optimizer, loss=Huber())

      history = model.fit(
        X_train_reshaped, y_train,
        epochs=200,
        verbose=0,
        batch_size=32,
        validation_split=0.2,
        callbacks=[
            EarlyStopping(patience=15, restore_best_weights=True),
            LearningRateScheduler(lr_scheduler)
        ]
      )
      for index, row in test_data.iterrows():
        df = pd.DataFrame([row])
        # т.к. все кроме первой строки будут иметь пропуски в лагах, то перед предсказанием устанавливается лаг равный -n предсказанию
        for col in df.columns[df.isnull().any()].tolist():
          df[col] = predictions[-rev_lags[col]]
        # и делается нормализация
        df[valid_columns] = scaler.transform(df[valid_columns])
        prediction = model.predict(df.drop('CLOSE', axis=1).to_numpy().reshape((1, 1, features)), verbose=0)[0]
        pred_df = copy.deepcopy(df)
        pred_df['CLOSE'] = prediction[0]
        # записываются ошибки и предсказание
        df[valid_columns] = scaler.inverse_transform(df[valid_columns])
        pred_df[valid_columns] = scaler.inverse_transform(pred_df[valid_columns])
        prediction_metric = metrics(np.array([df['CLOSE']]), np.array([pred_df['CLOSE']]))
        metric_scores.append({ 'rmse': prediction_metric[0], 'mape': prediction_metric[1] })
        predictions.append(pred_df.iloc[0, 0])
      data['predictions'] = predictions
      data['metric_scores'] = metric_scores
      data['importances'] = np.array([])
      data['time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      data['best_params'] = { 'timesteps': timesteps }
      data['best_model'] = model
      sum_first_10_mape = sum([metric_score['mape'] for metric_score in metric_scores][:10])
      if sum_first_10_mape < best_first_10_mape_sum:
        best_first_10_mape_sum = sum_first_10_mape
        best_data = copy.deepcopy(data)
    upload_models_data_to_s3(secid, best_data['name'], best_data)

fit_secids_from_s3('preprocessed_data/')

# secid = 'SBER'
# model_name = 'lstm'
# def download_models_data_from_s3(secid, model_name):
#     key = f'predictions/{secid}/{model_name}.pkl'
#     response = s3_client.get_object(Bucket=BUCKET, Key=key)
#     if response['ResponseMetadata']['HTTPStatusCode'] == 200:
#         print(f"Успешное получение в {BUCKET}/{key}")
#         return pickle.loads(response['Body'].read())
#     else:
#         print(f"Ошибка при получение: {response['ResponseMetadata']['HTTPStatusCode']}")

# fitted_model = download_models_data_from_s3(secid, model_name)
# print(fitted_model)
# metric_scores = fitted_model['metric_scores']
# sum([metric_score['mape'] for metric_score in metric_scores][:10])