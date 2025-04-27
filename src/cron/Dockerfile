FROM ubuntu:24.04

RUN apt-get update && apt-get -y install cron python3 python3-pip python3-venv

RUN python3 -m venv /usr/src/app/venv

COPY requirements.txt .
COPY daytime.py /usr/src/app/daytime.py

RUN echo "* * * * * root /usr/src/app/venv/bin/python /usr/src/app/daytime.py >> /var/log/cron.log 2>&1" >> /etc/crontab

RUN /usr/src/app/venv/bin/pip3 install --upgrade pip

RUN /usr/src/app/venv/bin/pip3 install --no-cache-dir -r requirements.txt

CMD ["sh", "-c", "/usr/src/app/venv/bin/python /usr/src/app/daytime.py >> /var/log/cron.log 2>&1 & cron -f -l 2"]
