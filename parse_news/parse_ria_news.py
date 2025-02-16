import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta, date
import re
import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from traceback import format_exc
import json
import traceback
import time
import pytz

# os.environ['AWS_ACCESS_KEY_ID'] = <access_key_id> 
# os.environ['AWS_SECRET_ACCESS_KEY'] = <secret_access_key>


BUCKET = 'russian-news'

access_key = os.getenv('AWS_ACCESS_KEY_ID')
secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
endpoint_url = 'https://storage.yandexcloud.net'

# Создание клиента S3
s3_client = boto3.client('s3',
                         region_name='ru-central1',
                         aws_access_key_id=access_key,
                         aws_secret_access_key=secret_key,
                         endpoint_url=endpoint_url)


def upload_object_to_s3(date, body):
    key = f'ria/{date}.pkl'
    response = s3_client.put_object(Bucket=BUCKET, Key=key, Body=body)
    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        print(f"Успешно сохранен в {BUCKET}/{key}")
    else:
        print(f"Ошибка при сохранении: {response['ResponseMetadata']['HTTPStatusCode']}")


def download_links_from_s3(date):
    key = f'ria/{date}.pkl'
    response = s3_client.get_object(Bucket=BUCKET, Key=key)
    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        print(f"Успешно получен из {BUCKET}/{key}")
    else:
        print(f"Ошибка при получении: {response['ResponseMetadata']['HTTPStatusCode']}")
    return json.loads(response['Body'].read())


def download_ria_news(start_date = datetime.strptime('2001-10-16', '%Y-%m-%d').date(), end_date = datetime.now(pytz.timezone('Europe/Moscow')).date()):
    if end_date > datetime.now(pytz.timezone('Europe/Moscow')).date():
        return print('Дата окончания в будущем')
    if start_date > end_date:
        return print('Дата начала больше окончания')

    directories = set()
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET, Prefix='ria')
        for context in response['Contents']:
            print(context['Key'])
        for page in paginator.paginate(Bucket=BUCKET, Delimiter='/ria'):
            for prefix in page.get('CommonPrefixes', []):
                directories.add(prefix.get('Prefix'))
    except NoCredentialsError:
        print("Ошибка: Неверные учетные данные.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
    return directories
    delta = timedelta(days=1)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    find_date = end_date
    wait_time = 10
    while (find_date >= start_date):
        last_id = -1
        news_links = set()
        print(find_date)
        reg_date = re.sub('-', '', str(find_date))
        url = f'https://ria.ru/{reg_date}/'
        while True:
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                divs = soup.find_all('div', class_='list-item__content')

                for div in divs:
                    links = div.find_all('a', href=True)
                    for link in links:
                        last_href = link['href']
                        if re.search(f'^https://ria.ru/{reg_date}/.+.*', last_href):
                            news_links.add(last_href)
                            href = last_href
                        else:
                            print(f'Плохие ссылки {last_href}')
            elif response.status_code == 429:
                print(f"По {url} получили {response.status_code} ждем секунду")
                time.sleep(wait_time)
                continue
            else:
                print(f"Ошибка при получении данных с {url}: {response.status_code}")
            next_date = soup.find_all('div', class_='list-item__info-item', attrs={'data-type': 'date'})
            if len(next_date) == 0:
                break
            date_time = re.sub(':', '', next_date[-1].text[-5:]) + '59'
            id = re.findall(r'(\d+)(?=\.html)', href)[-1]
            if id == last_id:
                break
            else:
                last_id = id
            url = f'https://ria.ru/services/{reg_date}/more.html?id={id}&date={reg_date}T{date_time}'
        print(f'Число сохраняемых ссылок: {len(news_links)}')
        try:
            upload_object_to_s3(find_date, json.dumps(list(news_links)))
        except ClientError as e:
            print(f"Произошла ошибка: {e.response['Error']['Message']}")
        except Exception as e:
            error_message = f"Неизвестная ошибка: {str(e)}"
            error_context = traceback.format_exc()
            print(f"{error_message}\nКонтекст ошибки:\n{error_context}")
        find_date -= delta

print(download_ria_news(end_date=datetime.strptime('2024-01-06', '%Y-%m-%d').date()))

import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import requests
from bs4 import BeautifulSoup
import os

# Настройки Yandex Object Storage
YANDEX_STORAGE_ENDPOINT = 'https://storage.yandexcloud.net'
SOURCE_BUCKET_NAME = 'russian-news'
TARGET_BUCKET_NAME = 'parsed-russian-news'  # Укажите имя вашего целевого бакета

# Инициализация клиента S3 для работы с Yandex Object Storage
session = boto3.Session(
    aws_access_key_id='YOUR_ACCESS_KEY_ID',  # Замените на ваш Access Key ID
    aws_secret_access_key='YOUR_SECRET_ACCESS_KEY'  # Замените на ваш Secret Access Key
)

s3_client = session.client('s3', endpoint_url=YANDEX_STORAGE_ENDPOINT)

def get_links_for_day(day_folder):
    """Получает список ссылок из файла для указанного дня."""
    try:
        response = s3_client.get_object(Bucket=SOURCE_BUCKET_NAME, Key=f'ria/{day_folder}.pkl')
        return json.loads(response['Body'].read())
    except Exception as e:
        print(f"Ошибка при чтении файла для дня {day_folder}: {e}")
        return []

def save_to_s3(content, day_folder):
    """Сохраняет контент в целевой бакет S3."""
    try:
        key = f'ria/{day_folder[:4]}/{day_folder}.pkl'
        response = s3_client.get_object(Bucket=TARGET_BUCKET_NAME, Key=key)
        data = json.loads(response['Body'].read()).append(content)
        s3_client.put_object(
            Bucket=TARGET_BUCKET_NAME,
            Key=key,
            Body=json.dumps(data)
        )
        print(f"Файл {day_folder}.pkl сохранён успешно.")
    except (NoCredentialsError, PartialCredentialsError) as e:
        print("Ошибка с учётными данными для S3:", e)
    except Exception as e:
        print(f"Ошибка при записи в целевой бакет: {e}")

def save_start_day_to_s3(content, day_folder):
    """Сохраняет контент в целевой бакет S3."""
    try:
        s3_client.put_object(
            Bucket=TARGET_BUCKET_NAME,
            Key=f'ria/{day_folder[:4]}/{day_folder}.pkl',
            Body=json.dumps(content)
        )
        print(f"Файл {day_folder}.pkl сохранён успешно.")
    except (NoCredentialsError, PartialCredentialsError) as e:
        print("Ошибка с учётными данными для S3:", e)
    except Exception as e:
        print(f"Ошибка при записи в целевой бакет: {e}")

def process_day(day_folder):
    """Обрабатывает все ссылки за указанный день."""
    links = get_links_for_day(day_folder)
    save_start_day_to_s3([], day_folder, )
    for link in links:
        content = parse_page(link)
        if content:
            filename = os.path.basename(link)
            save_to_s3(content, day_folder, filename)

def main():
    # Предполагается, что папки с днями именуются в формате YYYY-MM-DD
    days = ['2023-10-01', '2023-10-02']  # Укажите список дней для обработки

    for day in days:
        print(f"Обработка дня: {day}")
        process_day(day)

# текст article__text 
# текст в кавычках article__quote-text
# дата article__info-date в a
# тайтлы article__title и article__second-title
# анонс article__announce-text

# убирать strong и a, p, div, h1, h2, h3, h4, h5, h6

def clean_text(text):
    # Удаляем лишние пробелы и переводы строк
    return ' '.join(text.split()).strip()

def parse_page(url):
    # Получаем содержимое страницы
    response = requests.get(url)
    while response.status_code == 429:
        print(f"По {url} получили {response.status_code} ждем 10 секунд")
        time.sleep(10)
        response = requests.get(url)

    soup = BeautifulSoup(response.content, 'html.parser')

    # Извлечение даты
    date_element = soup.find('div', class_='article__info-date')
    date = clean_text(date_element.find('a').get_text()[:5]) if date_element else None

    # Извлечение заголовков
    titles = []
    for title_class in ['article__title', 'article__second-title']:
        elements = soup.find_all(class_=title_class)
        for element in elements:
            # Удаляем теги strong и a
            for tag in element(['strong', 'a']):
                tag.decompose()
            titles.append(clean_text(element.get_text()))

    # Извлечение текста
    texts = []
    text_classes = ['article__text', 'article__quote-text', 'article__announce-text']
    for text_class in text_classes:
        elements = soup.find_all(class_=text_class)
        for element in elements:
            # Удаляем теги p, div, h1, h2, h3, h4, h5, h6
            for tag in element(['a', 'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                tag.decompose()
            texts.append(clean_text(element.get_text()))

    # Структурированные данные
    page_data = {
        'date': date,
        'titles': titles,
        'texts': texts
    }

    return page_data
