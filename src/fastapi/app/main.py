from loguru import logger
from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from schemas import Item, ItemCreate
from crud import create_item, get_items
from database import SessionLocal
from pika import ConnectionParameters, BlockingConnection, PlainCredentials
from botocore.exceptions import ClientError, NoCredentialsError
from fastapi.responses import FileResponse
import time
import os
import redis
import asyncio
import boto3
import pickle
import json
import pandas as pd
from dotenv import load_dotenv
import numpy as np

load_dotenv()

f = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {name} | {level} | {message}"
# Настройка логгирования
logger.add(
    "logs/file_{time:YYYY-MM-DD}.log",
    format=f,
    rotation="1 day",
    retention="180 days",
    compression="zip",
    enqueue=True,
    backtrace=True,
    diagnose=True
)

connection_params = ConnectionParameters('rabbitmq', 5672, '/', PlainCredentials(os.getenv("RABBITMQ_DEFAULT_USER"), os.getenv("RABBITMQ_DEFAULT_PASS")))

app = FastAPI()

redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))
client = redis.StrictRedis(host=redis_host, port=redis_port, decode_responses=True)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_url = request.url.path
    if request.headers.get("X-Forwarded-For", None) is None:
        return RedirectResponse(url=f'https://nginx{request_url}', status_code=308)

    client_ip = request.client.host
    request_method = request.method

    if request.method == "POST":
        request_params = await request.json()
    else:
        request_params = request.query_params

    request = f'{request_method} {request_url}, параметры "{request_params}"'
    logger.info(f'Запрос от {client_ip}: {request}')

    response = await call_next(request)

    answer = f'{response.status_code} для {request_method} {request_url}'
    logger.info(f"Ответ статус: {answer}")

    return response


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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


def download_models_data_from_s3(secid, model_name):
    key = f'predictions/{secid}/{model_name}.pkl'
    response = s3_client.get_object(Bucket=BUCKET, Key=key)
    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        logger.info(f"Успешное получение в {BUCKET}/{key}")
        return pickle.loads(response['Body'].read())
    else:
        logger.info(f"Ошибка при получение: {response['ResponseMetadata']['HTTPStatusCode']}")


def get_best_models(secid):
    key = f'predictions/{secid}/best_models.pkl'
    response = s3_client.get_object(Bucket=BUCKET, Key=key)
    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        logger.info(f"Успешное получение в {BUCKET}/{key}")
        return pickle.loads(response['Body'].read())
    else:
        logger.info(f"Ошибка при получение: {response['ResponseMetadata']['HTTPStatusCode']}")


def download_mean_models_data_from_s3(model_name, duration):
    mape = None
    rmse = None
    key = f'predictions/{duration}{model_name}_mean_mape.pkl'
    response = s3_client.get_object(Bucket=BUCKET, Key=key)
    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        logger.info(f"Успешное получение в {BUCKET}/{key}")
        mape = json.loads(response['Body'].read())
    else:
        logger.info(f"Ошибка при получение: {response['ResponseMetadata']['HTTPStatusCode']}")

    key = f'predictions/{duration}{model_name}_mean_rmse.pkl'
    response = s3_client.get_object(Bucket=BUCKET, Key=key)
    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        logger.info(f"Успешное получение в {BUCKET}/{key}")
        rmse = json.loads(response['Body'].read())
    else:
        logger.info(f"Ошибка при получение: {response['ResponseMetadata']['HTTPStatusCode']}")

    return mape, rmse


GROUPED_IMAGES = {
    'Весь график': {'': 'cost'},
    'Валидация и предсказания': { 'неделя': 'val_cost_week', 'месяц': 'val_cost_month', 'год': 'val_cost_year', 'максимум': 'val_cost_maximum'},
    'MAPE ошибка': { 'неделя': 'MAPE_errors_week', 'месяц': 'MAPE_errors_month', 'год': 'MAPE_errors_year', 'максимум': 'MAPE_errors_maximum'},
    'RMSE ошибка': { 'неделя': 'RMSE_errors_week', 'месяц': 'RMSE_errors_month', 'год': 'RMSE_errors_year', 'максимум': 'RMSE_errors_maximum'}
}

def get_images(secid, model):
    grouped_images = dict()
    for category, images in GROUPED_IMAGES.items():
        if category not in grouped_images:
            grouped_images[category] = dict()
        for name, image in images.items():
            grouped_images[category][name] = f'predictions/{secid}/images/{model}_{image}.png'
    return grouped_images

GROUPED_MEAN_IMAGES = {
    'MAPE ошибка': { 'неделя': 'MAPE_mean_errors_week', 'месяц': 'MAPE_mean_errors_month', 'год': 'MAPE_mean_errors_year', 'максимум': 'MAPE_mean_errors_maximum'},
    'RMSE ошибка': { 'неделя': 'RMSE_mean_errors_week', 'месяц': 'RMSE_mean_errors_month', 'год': 'RMSE_mean_errors_year', 'максимум': 'RMSE_mean_errors_maximum'}
}

def get_mean_images(model, duration):
    grouped_images = dict()
    for category, images in GROUPED_MEAN_IMAGES.items():
        if category not in grouped_images:
            grouped_images[category] = dict()
        for name, image in images.items():
            logger.info(category)
            grouped_images[category][name] = f'predictions/images/{duration}{model}_{image}.png'
    return grouped_images


DAYS = [1, 5, 10, 21, 62, 124, 247]

def get_top_models_images(duration):
    grouped_images = list()
    for day in DAYS:
        grouped_images.append(f'predictions/images/{duration}mean_top_models_{day}.png')
    return grouped_images


def list_directories(s3_client):
    directories = set()
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=f'{BUCKET}', Delimiter='/', Prefix='predictions/'):
            for prefix in page.get('CommonPrefixes', []):
                dir = prefix.get('Prefix').split('/')[-2]
                directories.add(dir)
    except NoCredentialsError:
        print("Ошибка: Неверные учетные данные.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
    return sorted(directories)

SECIDS = list_directories(s3_client)
if 'images' in SECIDS:
    SECIDS.remove('images')

MODELS = ['ridge', 'random_forest', 'xgboost', 'lstm', 'sarimax', 'ridge_with_news', 'random_forest_with_news', 'xgboost_with_news', 'lstm_with_news']

@app.get("/secids/")
async def read_secids():
    return SECIDS


@app.get("/models/")
async def read_models():
    return MODELS


@app.get("/predict/{model}/{secid}/")
async def predict(model: str, secid: str):
    """
    Получение обучаемых данных, предиктов и реальных значений
    """
    # if model not in MODELS:
    #     raise HTTPException(status_code=400, detail="Invalid model")
    
    # if secid not in SECIDS:
    #     raise HTTPException(status_code=400, detail="Invalid secid")
    
    data = download_models_data_from_s3(secid, model)
    predictions = np.float64(data['predictions'])
    rmse = [metric['rmse'] for metric in data['metric_scores']]
    mape = [metric['mape'] for metric in data['metric_scores']]
    data_frame = pd.DataFrame({'День': list(range(1, len(predictions) + 1)), 'Прогноз': predictions, 'RMSE': rmse, 'MAPE': mape })
    data_frame.set_index('День', inplace=True)
    grouped_images = get_images(secid, model)
    best_models = get_best_models(secid)
    best_models_text = list()
    logger.info(best_models)
    for index in range(len(best_models)):
        best_models_text.append(f'День {DAYS[index]}: {best_models[index][0]} - {round(best_models[index][1], 3)}')
    return {
        "data_frame": data_frame,
        "grouped_images": grouped_images,
        "best_models": best_models_text
    }


@app.get("/predict_mean/{model}/{duration}")
async def predict_mean(model: str, duration: str):
    """
    Получение средних результатов для кажодой из модели.
    """
    # if model not in MODELS:
    #     raise HTTPException(status_code=400, detail="Invalid model")
    if duration == 'all':
        duration = ''
    else:
        duration = 'five_years_'
    mape, rmse = download_mean_models_data_from_s3(model, duration)
    rmse = [value if value != float('inf') and value != float('-inf') else 9e200 for value in rmse]
    mean_data_frame = pd.DataFrame({'День': list(range(1, len(mape) + 1)), 'RMSE': rmse, 'MAPE': mape })
    mean_data_frame.set_index('День', inplace=True)
    grouped_mean_images = get_mean_images(model, duration)
    return {
        "mean_data_frame": mean_data_frame,
        "grouped_mean_images": grouped_mean_images
    }


@app.get("/top_models/{duration}")
async def top_models(duration: str):
    """
    Получение лучших моделей на различные горизонты дней.
    """
    # if model not in MODELS:
    #     raise HTTPException(status_code=400, detail="Invalid model")
    if duration == 'all':
        duration = ''
    else:
        duration = 'five_years_'
    top_models_images = get_top_models_images(duration)
    return {
        "top_models_images": top_models_images
    }


# @app.post("/stats/{model}/{symbol}")
# async def stats(model: str, symbol: str):
#     """
#     Возвращает статистику и графику
#     """
#     if model not in ["ModelA", "ModelB"]:
#         raise HTTPException(status_code=400, detail="Invalid model")
    
#     if symbol not in SYMBOLS:
#         raise HTTPException(status_code=400, detail="Invalid symbol")
    
#     # Здесь симулируем возврат изображений
#     images = [
#         "https://via.placeholder.com/300x200.png",
#         "https://via.placeholder.com/300x200.png"
#     ]
#     return {"images": images, "best_model": "Best Stats"}


# @app.get("/")
# async def read_root():
#     return {"Hello": "World"}


# @app.post("/items/", response_model=Item)
# async def create_items(item: ItemCreate, db: Session = Depends(get_db)):
#     return create_item(db=db, item=item)


# @app.get("/items/", response_model=list[Item])
# async def read_items(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
#     items = get_items(db, skip=skip, limit=limit)
#     return items


# @app.get("/cron/")
# async def read_cron_result():
#     value = client.get('rabbit:cron')
#     if value is not None:
#         return { 'messages_redis': value }
#     with BlockingConnection(connection_params) as conn:
#         with conn.channel() as ch:
#             ch.queue_declare(queue='daytime')
#             method_frame, header_frame, body = ch.basic_get(queue='daytime')
#             if body is None:
#                 await asyncio.sleep(0.5)
#                 method_frame, header_frame, body = ch.basic_get(queue='daytime')
#             if body is not None:
#                 client.set('rabbit:cron', body, ex=10)
#     return { 'messages_rabbit': body }

# @app.exception_handler(HTTPException)
# async def http_exception_handler(request, exc):
#     return JSONResponse(
#         status_code=exc.status_code,
#         content={"message": f"Ошибка: {exc.detail}"},
#     )
