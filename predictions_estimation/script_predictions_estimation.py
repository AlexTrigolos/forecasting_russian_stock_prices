import copy
import os
import boto3
import traceback
import io
import json
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from botocore.exceptions import ClientError, NoCredentialsError
from traceback import format_exc
from io import BytesIO

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

def upload_object_to_s3(key, body):
    response = s3_client.put_object(Bucket=BUCKET, Key=key, Body=body)
    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        print(f"Успешно сохранен в {BUCKET}/{key}")
    else:
        print(f"Ошибка при сохранении: {response['ResponseMetadata']['HTTPStatusCode']}")

def download_models_data_from_s3(secid, model_name):
    key = f'predictions/{secid}/{model_name}.pkl'
    print(key)
    response = s3_client.get_object(Bucket=BUCKET, Key=key)
    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        print(f"Успешное получение в {BUCKET}/{key}")
        return pickle.loads(response['Body'].read())
    else:
        print(f"Ошибка при получение: {response['ResponseMetadata']['HTTPStatusCode']}")

def download_data_frame_from_s3(secid):
    key = f'preprocessed_data/secids/{secid}/{secid}_data_frame.pkl'
    response = s3_client.get_object(Bucket=BUCKET, Key=key)
    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        print(f"Успешное получение в {BUCKET}/{key}")
        return pickle.loads(response['Body'].read())
    else:
        print(f"Ошибка при получение: {response['ResponseMetadata']['HTTPStatusCode']}")


def list_directories(s3_client):
    directories = set()
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=BUCKET, Delimiter='/', Prefix='predictions/'):
            for prefix in page.get('CommonPrefixes', []):
                directories.add(prefix.get('Prefix').split('/')[-2])
    except NoCredentialsError:
        print("Ошибка: Неверные учетные данные.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
    return directories

secids = sorted(list(list_directories(s3_client)))
secids.remove('images')

def save_graph_cost(data_frame, fitted_model, secid, model_name):
  plt.figure(figsize=(10, 6))

  train = data_frame.iloc[:len(data_frame) - len(fitted_model['predictions'])]
  val = data_frame.iloc[len(data_frame) - len(fitted_model['predictions']):]
  # Тренировочные данные
  plt.plot(pd.to_datetime(train['TRADEDATE']), train['CLOSE'], ':', color='blue', label='Тренировочные данные')

  # Валидационные данные
  plt.plot(pd.to_datetime(val['TRADEDATE']), val['CLOSE'], ':', color='orange', label='Валидационные данные')

  # Предсказания
  plt.plot(pd.to_datetime(val['TRADEDATE']), fitted_model['predictions'], ':', color='red', label='Предсказания')

  # Настройка графика
  plt.title(f'График тренировочных и валидационных данных с предсказаниями для {secid}')
  plt.xlabel('День')
  plt.ylabel('Стоимость')
  plt.legend()
  plt.grid(True)

  # Сохранить график
  img_buffer = BytesIO()
  plt.savefig(img_buffer, format='png')
  img_buffer.seek(0)
  plt.close()
  s3_client.put_object(Bucket=BUCKET, Key= f'predictions/{secid}/images/{model_name}_cost.png', Body=img_buffer)

def save_graph_val_cost(data_frame, fitted_model, secid, model_name):
  val = data_frame.iloc[len(data_frame) - len(fitted_model['predictions']):]

  for name, days_count in { 'week': 5, 'month': 21, 'year': 247, 'maximum': 999999999}.items():
    if len(val['CLOSE']) < days_count and name != 'maximum':
      continue
    limit = min(len(val['CLOSE']), days_count)
    plt.figure(figsize=(10, 6))

    # Валидационные данные
    plt.plot(pd.to_datetime(val['TRADEDATE'][:limit]), val['CLOSE'][:limit], ':', color='orange', label='Валидационные данные')

    # Предсказания
    plt.plot(pd.to_datetime(val['TRADEDATE'][:limit]), fitted_model['predictions'][:limit], ':', color='red', label='Предсказания')

    # Настройка графика
    plt.title(f'График валидационных данных с предсказаниями для {secid}')
    plt.xlabel('День')
    plt.ylabel('Стоимость')
    plt.legend()
    plt.grid(True)

    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()
    s3_client.put_object(Bucket=BUCKET, Key= f'predictions/{secid}/images/{model_name}_val_cost_{name}.png', Body=img_buffer)

def save_graph_errors(data_frame, fitted_model, secid, model_name, error, error_name):
  val = data_frame.iloc[len(data_frame) - len(fitted_model['predictions']):]

  for name, days_count in { 'week': 5, 'month': 21, 'year': 247, 'maximum': 999999999}.items():
    if len(val['CLOSE']) < days_count and name != 'maximum':
      continue
    limit = min(len(val['CLOSE']), days_count)
    plt.figure(figsize=(10, 6))
    plt.plot(pd.to_datetime(val['TRADEDATE'][:limit]), error[:limit], ':', color='red', label=error_name)

    # Настройка графика
    plt.title(f'График ошибок в предсказаниях для {secid}')
    plt.xlabel('День')
    plt.ylabel('Ошибка')
    plt.legend()
    plt.grid(True)

    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()
    s3_client.put_object(Bucket=BUCKET, Key= f'predictions/{secid}/images/{model_name}_{error_name}_errors_{name}.png', Body=img_buffer)

def save_graph_mean_errors(model_name, error, error_name):
  for name, days_count in { 'week': 5, 'month': 21, 'year': 247, 'maximum': 999999999}.items():
    if len(error) < days_count and name != 'maximum':
      continue
    limit = min(len(error), days_count)

    plt.figure(figsize=(10, 6))

    plt.plot(list(range(1, limit + 1)), error[:limit], ':', color='red', label=error_name)

    # Настройка графика
    plt.title('График средних ошибок в предсказаниях')
    plt.xlabel('День')
    plt.ylabel('Ошибка')
    plt.legend()
    plt.grid(True)

    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()
    s3_client.put_object(Bucket=BUCKET, Key= f'predictions/images/{model_name}_{error_name}_mean_errors_{name}.png', Body=img_buffer)

def save_graph_best_worse(error, error_name, tonality):
  for name, days_count in { 'week': 5, 'month': 21, 'year': 247, 'maximum': 999999999}.items():
    if len(error) < days_count and name != 'maximum':
      continue
    limit = min(len(error), days_count)
    plt.figure(figsize=(10, 6))
    plt.plot(list(range(1, limit + 1)), error[:limit], ':', color='red', label=error_name)

    # Настройка графика
    plt.title(f'График {"лучших" if tonality == "best" else "худших"} ошибок в предсказаниях')
    plt.xlabel('День')
    plt.ylabel('Ошибка')
    plt.legend()
    plt.grid(True)

    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()
    s3_client.put_object(Bucket=BUCKET, Key= f'predictions/images/{tonality}_{error_name}_errors_{name}.png', Body=img_buffer)

def upload_errors_data(data, error_name):
    json_data = json.dumps(data)
    error_name_file = f'predictions/{error_name}.pkl'
    upload_object_to_s3(error_name_file, json_data)

model_names = ['random_forest', 'ridge']
best_rmse = list()
best_mape = list()
worse_rmse = list()
worse_mape = list()
for model_name in model_names:
  rmse_data = list()
  mape_data = list()
  for secid in secids:
    fitted_model = download_models_data_from_s3(secid, model_name)
    data_frame = download_data_frame_from_s3(secid)
    save_graph_cost(data_frame, fitted_model, secid, model_name)
    save_graph_val_cost(data_frame, fitted_model, secid, model_name)
    rmse = list()
    mape = list()
    for metrics in fitted_model['metric_scores']:
      rmse.append(metrics['rmse'])
      mape.append(metrics['mape'])
    save_graph_errors(data_frame, fitted_model, secid, model_name, rmse, 'RMSE')
    save_graph_errors(data_frame, fitted_model, secid, model_name, mape, 'MAPE')
    for indx in range(len(rmse)):
      if len(rmse_data) > indx:
        rmse_data[indx] += rmse[indx]
      else:
        rmse_data.append(rmse[indx])

      if len(best_rmse) > indx:
        if best_rmse[indx]['value'] > rmse[indx]:
          best_rmse[indx] = { 'value': rmse[indx], 'secid': secid, 'model': model_name }
      else:
        best_rmse.append({ 'value': rmse[indx], 'secid': secid, 'model': model_name })

      if len(worse_rmse) > indx:
        if worse_rmse[indx]['value'] < rmse[indx]:
          worse_rmse[indx] = { 'value': rmse[indx], 'secid': secid, 'model': model_name }
      else:
        worse_rmse.append({ 'value': rmse[indx], 'secid': secid, 'model': model_name })

      if len(mape_data) > indx:
        mape_data[indx] += mape[indx]
      else:
        mape_data.append(mape[indx])

      if len(best_mape) > indx:
        if best_mape[indx]['value'] > mape[indx]:
          best_mape[indx] = { 'value': mape[indx], 'secid': secid, 'model': model_name }
      else:
        best_mape.append({ 'value': mape[indx], 'secid': secid, 'model': model_name })

      if len(worse_mape) > indx:
        if worse_mape[indx]['value'] < mape[indx]:
          worse_mape[indx] = { 'value': mape[indx], 'secid': secid, 'model': model_name }
      else:
        worse_mape.append({ 'value': mape[indx], 'secid': secid, 'model': model_name })

  for indx in range(len(rmse_data)):
    rmse_data[indx] /= len(secids)

  for indx in range(len(mape_data)):
    mape_data[indx] /= len(secids)

  save_graph_mean_errors(model_name, rmse_data, 'RMSE')
  save_graph_mean_errors(model_name, mape_data, 'MAPE')

upload_errors_data(best_rmse, 'best_rmse')
upload_errors_data(best_mape, 'best_mape')
upload_errors_data(worse_rmse, 'worse_rmse')
upload_errors_data(worse_mape, 'worse_mape')
save_graph_best_worse([item['value'] for item in best_rmse], 'RMSE', 'best')
save_graph_best_worse([item['value'] for item in best_mape], 'MAPE', 'best')
save_graph_best_worse([item['value'] for item in worse_rmse], 'RMSE', 'worse')
save_graph_best_worse([item['value'] for item in worse_mape], 'MAPE', 'worse')
