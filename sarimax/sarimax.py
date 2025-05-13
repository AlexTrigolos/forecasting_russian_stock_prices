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

from sklearn.preprocessing import MinMaxScaler
import copy
import datetime
import pickle
import statsmodels as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools
# from pmdarima import auto_arima

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

def download_data_from_s3(dir, secids=None):
    data = {}
    try:
        directories = download_secid_names(dir)
        for secid in directories:
            if (secids is None or secid in secids):
                if secids is not None:
                    secids.remove(secid)
                data[secid] = download_info_from_s3(dir, secid)
                data[secid]['data_frame'] = download_data_frame_from_s3(dir, secid)
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
  metric_scores = []
  for index in range(len(y_true)):
    # корень из квадратичной ошибки, возвращает ошибку в тех же единицах, что и целевая переменная
    rmse_score = rmse(np.array([y_true.iloc[index]]), np.array([y_pred.iloc[index]]))
    # измеряет ошибку в процентах и позволяет легко интерпретировать результаты
    mape_score = mape(np.array([y_true.iloc[index]]), np.array([y_pred.iloc[index]]))
    metric_scores.append({ 'rmse': rmse_score, 'mape': mape_score })
  return metric_scores

data_frames = download_data_from_s3('preprocessed_data/')


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

secids = download_secid_names('preprocessed_data/')


def search_optimal_sarima(time_series, secid):
  ps = [0, 1, 2, 3]
  ds = [0, 1]
  qs = [0, 1, 2, 3]
  # Ps = [5, 21, 247]
  # Ds = [5, 21, 247]
  # Qs = [5, 21, 247]
  # ms = [5, 21, 247]
  Ps = [5]
  Ds = [5]
  Qs = [5]
  ms = [5]
  combinations = list(itertools.product(ps, ds, qs, Ps, Ds, Qs, ms))

  smallest_aic = float("inf")
  best_model_results = None
  optimal_order_param = optimal_seasonal_param = None

  for combination in combinations:
    print(combination, secid)
    sarima_model = SARIMAX(time_series['CLOSE'],
                           order=combination[:3],
                           seasonal_order=combination[3:],
                           enforce_stationarity=False,
                           enforce_invertibility=False)

    model_results = sarima_model.fit(disp=False)
    if model_results.aic < smallest_aic:
      smallest_aic = model_results.aic
      best_model_results = model_results
      optimal_order_param = combination[:3]
      optimal_seasonal_param = combination[3:]
  return best_model_results, { 'p': optimal_order_param[0], 'd': optimal_order_param[1], 'q': optimal_order_param[2],  'P': optimal_seasonal_param[0], 'D': optimal_seasonal_param[1], 'Q': optimal_seasonal_param[2], 'm': optimal_seasonal_param[3] }


for secid in secids:
  # Получаем данные по бумаге и удаляем дату
  secid_data = data_frames[secid]['data_frame'][['TRADEDATE', 'CLOSE']]
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
          'name': 'sarimax',
          'model': 'SARIMAX',
          'grid_params': {},
          'importances_name': ''
      }
  ]
  models_data = copy.deepcopy(base_models)
  for data in models_data:
    print(f"Обучение {data['model']} на {secid}")
    # обучение
    model_results, best_params = search_optimal_sarima(train, secid)
    # высчитываются предсказания и ошибки
    predictions = model_results.forecast(len(test))
    metric_scores = metrics(test['CLOSE'], predictions)
    # сохраняем необходимые данные вместе с обучением
    data['predictions'] = [prediction for prediction in predictions]
    data['metric_scores'] = metric_scores
    data['importances'] = np.array([])
    data['time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data['best_params'] = best_params
    data['best_model'] = 'SARIMAX'
    upload_models_data_to_s3(secid, data['name'], data)
