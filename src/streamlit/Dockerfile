# streamlit/Dockerfile
FROM python:3.9

WORKDIR /app

RUN pip install --upgrade pip

COPY app/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
