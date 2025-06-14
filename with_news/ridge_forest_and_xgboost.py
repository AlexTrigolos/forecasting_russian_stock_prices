import copy
import os
import boto3
import traceback
import io
import json
import xgboost

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from botocore.exceptions import ClientError, NoCredentialsError
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import copy
import datetime
import pickle

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

def download_news_info_from_s3(dir, secid):
    key = f'{dir}secids/{secid}/news_info.pkl'
    response = download_object_from_s3(key)
    return json.loads(response)

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
            if (secids is None or secid in secids):
                if secids is not None:
                    secids.remove(secid)
                try:
                    news_info = download_news_info_from_s3(dir, secid)
                except Exception:
                    news_info = {}
                fit_secid(secid, download_data_frame_from_s3(dir, secid), news_info)
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

def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def metrics(y_true, y_pred):
  # корень из квадратичной ошибки, возвращает ошибку в тех же единицах, что и целевая переменная
  rmse_score = rmse(y_true, y_pred)
  # измеряет ошибку в процентах и позволяет легко интерпретировать результаты
  mape_score = mape(y_true, y_pred)
  return tuple([rmse_score, mape_score])

# кастомная метрика для GridSearch
rmse_score = make_scorer(rmse, greater_is_better = False)

random_forest_grid_params = {
    'random_state': [42],
    'n_estimators': [50, 100, 200, 300, 400, 500],
    'max_features': ['log2', 'sqrt'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

ridge_grid_params = {
    'random_state': [42],
    'alpha': np.logspace(-4, 4, 10000),
    'fit_intercept': [True, False]
}

xgboost_grid_params = {
    'random_state': [42],
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.5, 0.7, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0],
    'gamma': [0, 1, 5]
}

base_models = [
    # {
    #     'name': 'ridge_with_news',
    #     'model': Ridge(),
    #     'grid_params': ridge_grid_params,
    #     'importances_name': 'coef_'
    # },
    # {
    #   'name': 'random_forest_with_news',
    #   'model': RandomForestRegressor(),
    #   'grid_params': random_forest_grid_params,
    #   'importances_name': 'feature_importances_'
    # },
    {
      'name': 'xgboost_with_news',
      'model': XGBRegressor(),
      'grid_params': xgboost_grid_params,
      'importances_name': 'feature_importances_'
    }
]

lags = { 1: 'lag_1', 2: 'lag_2', 3: 'lag_3', 4: 'lag_4', 5: 'lag_week', 10: 'lag_2_weeks',
        21: 'lag_month', 62: 'lag_3_months', 124: 'lag_half_year', 247: 'lag_year',
        371: 'lag_year_with_half', 495: 'lag_2_years', 742: 'lag_3_years' }
rev_lags = { 'lag_1': 1, 'lag_2': 2, 'lag_3': 3, 'lag_4': 4, 'lag_week': 5, 'lag_2_weeks': 10,
            'lag_month': 21, 'lag_3_months': 62, 'lag_half_year': 124, 'lag_year': 247,
            'lag_year_with_half': 371, 'lag_2_years': 495, 'lag_3_years': 742 }

def upload_models_data_to_s3(secid, model_name, body):
    key = f'predictions/{secid}/{model_name}.pkl'
    response = s3_client.put_object(Bucket=BUCKET, Key=key, Body=pickle.dumps(body))
    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        print(f"Успешно сохранен в {BUCKET}/{key}")
    else:
        print(f"Ошибка при сохранении: {response['ResponseMetadata']['HTTPStatusCode']}")

# Делаем GridSearch используя разделение для временных рядов и возвращаем лучшую модель
def fit_grid_search_with_cross_val(data, model, param_grid, target):
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, verbose=0, scoring=rmse_score)
    grid_search.fit(data.drop(target, axis=1), data[target])
    print(f'Лучшие параметры: {grid_search.best_params_}')
    return grid_search.best_estimator_, grid_search.best_params_


def fit_secid(secid, data_frame, news_info):
  # Получаем данные по бумаге
  secid_data = data_frame[['TRADEDATE', 'CLOSE']]
  # довавляем новостные колонки
  for column, value in { 'min_importance': 0, 'max_importance': 0, 'min_importance_hour': 14, 'min_importance_minute': 0, 'max_importance_hour': 14, 'max_importance_minute': 0, 'count_news': 1, 'mean_importance': 0 }.items():
    secid_data[column] = value

  for index, row in secid_data.iterrows():
    trade_date = row['TRADEDATE'].date()
    if str(trade_date) in news_info:
        for column, value in news_info[str(trade_date)].items():
            if str(column) != 'min_importance_time' and str(column) != 'max_importance_time':
                secid_data.at[index, column] = value
            else:
                hour, minute = value.split(':')
                secid_data.at[index, column.replace('time', 'hour')] = int(hour)
                secid_data.at[index, column.replace('time', 'minute')] = int(minute)

  # удаляем дату
  subset = pd.to_datetime(secid_data['TRADEDATE'])
  secid_data = secid_data.drop('TRADEDATE', axis=1)

  # добавляем дату по отдельности
  secid_data.loc[:, 'year'] = subset.dt.year
  secid_data.loc[:, 'month'] = subset.dt.month
  secid_data.loc[:, 'day'] = subset.dt.day

  importance_columns = ['min_importance', 'max_importance', 'min_importance_hour', 'min_importance_minute', 'max_importance_hour', 'max_importance_minute', 'count_news', 'mean_importance']
  # Добавляем отступы по возможным корреляциям (очень сложно выбрать нормальные отсутпы по причине того, что торги на бирже не нормированы, есть праздники, переносы, блокировки торгов, переезд компаний и другое)
  # но в среднем интернет выдал 247 с хвостиком рабочих дней в году, что я уже пытался нормально разделить, например для месяца получается 21 торговый день, хоть дней примерно 30
  for lag_name, lag_num in rev_lags.items():
    secid_data[lag_name] = secid_data['CLOSE'].shift(lag_num)
    for importance_column in importance_columns:
        secid_data[f'{lag_name}_{importance_column}'] = secid_data[importance_column].shift(lag_num)

  # Далее убираются строки по лагам, которые не имеют данных, если данных остается меньше чем на 3 месяца или меньше 10% от начальных данных, то удаляется полностью колонка
  # Потому что может быть ситуация, что данных на 3 года и 2 месяца, и только 2 месяца будут иметь лаг в 3 года, а я не хочу удалять так много данных
  # Если же данных достаточно, то это будет самый большой лаг и удаляются строки, в которых по этому лагу пропуски. И переназначаем индексы
  lag_names = list(reversed(lags.values()))
  for lag in lag_names:
    if secid_data[lag].isnull().sum() > 0:
      temp = secid_data.dropna(subset=[lag])
      if (len(temp) < 62 and len(secid_data) >= 62) or len(temp) * 4 < len(secid_data) or len(temp) < 10:
        secid_data = secid_data.drop(columns=[lag])
        for importance_column in importance_columns:
            secid_data = secid_data.drop(columns=[f'{lag}_{importance_column}'])
      else:
        secid_data = temp
        break
  secid_data = secid_data.reset_index().drop('index', axis=1)
  if secid_data.shape[0] < 6:
    return

  for importance_column in importance_columns:
    secid_data[f'next_{importance_column}'] = secid_data[importance_column].shift(-1)

  secid_data = secid_data[:-1]

  # Разбиваем данные, в валидацию идет 20%
  train_size = int(len(secid_data) * 0.8)
  train, val = secid_data[:train_size], secid_data[train_size:]

  valid_columns = [column for column in list(reversed(lags.values())) if column in val.columns.tolist()]
  # Очищаем те данные, что в валидацию попали из валидационных данных (чтобы не пытаться использовать известные реальные целевые значения для обучения)
  for lag_name, lag_num in rev_lags.items():
    if lag_name in valid_columns:
      val.loc[val[lag_name] == val['CLOSE'].shift(lag_num), lag_name] = np.nan
      for importance_column in importance_columns:
          val.loc[val[f'{lag_name}_{importance_column}'] == val[importance_column].shift(lag_num), f'{lag_name}_{importance_column}'] = np.nan


  importance_data_cols = ['min_importance_hour', 'min_importance_minute', 'max_importance_hour', 'max_importance_minute', 'count_news']
  for importance_column in importance_data_cols:
    valid_columns.extend([f'{lag}_{importance_column}' for lag in rev_lags.keys() if lag in valid_columns])
  valid_columns.extend(importance_data_cols)

  # нормализуем лаги (не трогаем таргет и даты)
  scaler = MinMaxScaler()
  train.loc[:, valid_columns] = scaler.fit_transform(train[valid_columns])

  models_data = copy.deepcopy(base_models)
  for data in models_data:
    print(f"Обучение {data['model']} на {secid}")
    metric_scores = list()
    predictions = list()

    train_data = copy.deepcopy(train)
    val_data = copy.deepcopy(val)

    model = data['model']
    # обучаем модель
    target_columns = [f'next_{importance_column}' for importance_column in importance_columns] + ['CLOSE']
    best_estimator, best_params = fit_grid_search_with_cross_val(train_data, model, data['grid_params'], target_columns)
    # делаем валидацию построчно, чтоб предсказание использовать для следующих предсказаний
    for index, row in val_data.iterrows():
      df = pd.DataFrame([row])
      # т.к. все кроме первой строки будут иметь пропуски в лагах, то перед предсказанием устанавливается лаг равный -n предсказанию
      for col in df.columns[df.isnull().any()].tolist():
        for lag in reversed(rev_lags.keys()):
            if col.startswith(lag):
              if lag == col:
                df[col] = predictions[-rev_lags[lag]][-1]
              else:
                for index in range(len(importance_columns)):
                    if col.endswith(importance_columns[index]):
                        df[col] = predictions[-rev_lags[lag]][index]
              break
      # и делается нормализация
      df[valid_columns] = scaler.transform(df[valid_columns])
      prediction = best_estimator.predict(df.drop(target_columns, axis=1))
      # записываются ошибки и предсказание
      prediction_metric = metrics(np.array([df['CLOSE']]), np.array([prediction[0][-1]]))
      metric_scores.append({ 'rmse': prediction_metric[0], 'mape': prediction_metric[1] })
      predictions.append(prediction[0])
    # сохраняем необходимые данные вместе с обучением
    data['predictions'] = [prediction[-1] for prediction in predictions]
    data['metric_scores'] = metric_scores
    data['importances'] = getattr(best_estimator, data['importances_name'])
    data['time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data['best_params'] = best_params
    data['best_model'] = best_estimator
    upload_models_data_to_s3(secid, data['name'], data)

fit_secids_from_s3('preprocessed_data/')


