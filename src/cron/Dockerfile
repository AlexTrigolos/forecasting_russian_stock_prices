FROM ubuntu:24.04

RUN apt-get update && apt-get -y install cron python3 python3-pip python3-venv

WORKDIR /usr/src/app

RUN python3 -m venv /usr/src/app/venv

COPY requirements.txt .
COPY daytime.py /usr/src/app/daytime.py

RUN echo "* * * * * root . /root/project_env.sh; /usr/src/app/venv/bin/python /usr/src/app/daytime.py >> /var/log/cron.log 2>&1" >> /etc/crontab

RUN /usr/src/app/venv/bin/pip3 install --upgrade pip

RUN /usr/src/app/venv/bin/pip3 install --no-cache-dir -r requirements.txt

COPY entrypoint.sh /usr/src/app/entrypoint.sh

RUN chmod +x /usr/src/app/entrypoint.sh

WORKDIR .

ENTRYPOINT [ "/usr/src/app/entrypoint.sh" ]
# CMD ["sh", "-c", "(/usr/src/app/venv/bin/python /usr/src/app/daytime.py >> /var/log/cron.log 2>&1) && (printenv | sed 's/^\(.*\)$/export \1/g' > /root/project_env.sh) && (cron -f -l 2)"]
