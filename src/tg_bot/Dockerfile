# tg_bot/Dockerfile
FROM python:3.9

WORKDIR .

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "bot.py"]
