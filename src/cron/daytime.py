from loguru import logger
from pika import ConnectionParameters, BlockingConnection, PlainCredentials
from datetime import datetime
import pytz
import os
from dotenv import load_dotenv

load_dotenv()

f = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {name} | {level} | {message}"
# Настройка логгирования
logger.add(
    "cron/cron.log",
    format=f,
    rotation="1 day",
    retention="180 days",
    compression="zip",
    enqueue=True,
    backtrace=True,
    diagnose=True
)
connection_params = ConnectionParameters('rabbitmq', 5672, '/', PlainCredentials(os.getenv("RABBITMQ_DEFAULT_USER"), os.getenv("RABBITMQ_DEFAULT_PASS")))


def cron_daytime():
    data = datetime.now(pytz.timezone('Europe/Moscow')).strftime("%Y-%m-%d %H:%M:%S")
    logger.info(data)
    with BlockingConnection(connection_params) as conn:
        with conn.channel() as ch:
            ch.queue_declare(queue='daytime')
            ch.queue_purge(queue='daytime')
            ch.basic_publish(exchange='', routing_key='daytime', body=data)

cron_daytime()
