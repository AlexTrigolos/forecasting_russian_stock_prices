import os
import boto3
import json
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
    return response['Body'].read()

def download_model_info_from_s3(dir, secid, model):
    key = f'{dir}{secid}/{model}.pkl'
    response = download_object_from_s3(key)
    return pickle.loads(response)

def download_secid_names(dir):
    key = f'{dir}secid_names.pkl'
    return json.loads(download_object_from_s3(key))

secids = download_secid_names('preprocessed_data/')
missed = { 'ridge': [], 'random_forest': [], 'xgboost': [], 'lstm': [], 'sarimax': [], 'ridge_with_news': [], 'random_forest_with_news': [], 'xgboost_with_news': [], 'lstm_with_news': [] }
for model_name in ['ridge', 'random_forest', 'xgboost', 'lstm', 'sarimax', 'ridge_with_news', 'random_forest_with_news', 'xgboost_with_news', 'lstm_with_news']:
  for secid in secids:
    try:
      download_model_info_from_s3('predictions/', secid, model_name)
    except Exception as e:
      missed[model_name].append(secid)
print(missed)
print([{ model_name: len(secids) } for model_name, secids in missed.items()])