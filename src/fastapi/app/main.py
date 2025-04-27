from loguru import logger
from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from schemas import Item, ItemCreate
from crud import create_item, get_items
from database import SessionLocal
from pika import ConnectionParameters, BlockingConnection, PlainCredentials
import time
import os
import redis
import asyncio

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

connection_params = ConnectionParameters('rabbitmq', 5672, '/', PlainCredentials('guest', 'guest'))

app = FastAPI()

redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))
client = redis.StrictRedis(host=redis_host, port=redis_port, decode_responses=True)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_url = request.url.path
    if request.headers.get("X-Forwarded-For", None) is None:
        return RedirectResponse(url=f'http://nginx{request_url}', status_code=308)

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


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/items/", response_model=Item)
async def create_items(item: ItemCreate, db: Session = Depends(get_db)):
    return create_item(db=db, item=item)


@app.get("/items/", response_model=list[Item])
async def read_items(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    items = get_items(db, skip=skip, limit=limit)
    return items


@app.get("/cron/")
async def read_cron_result():
    value = client.get('rabbit:cron')
    if value is not None:
        return { 'messages_redis': value }
    with BlockingConnection(connection_params) as conn:
        with conn.channel() as ch:
            ch.queue_declare(queue='daytime')
            method_frame, header_frame, body = ch.basic_get(queue='daytime')
            if body is None:
                await asyncio.sleep(0.5)
                method_frame, header_frame, body = ch.basic_get(queue='daytime')
            if body is not None:
                client.set('rabbit:cron', body, ex=10)
    return { 'messages_rabbit': body }

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": f"Ошибка: {exc.detail}"},
    )
