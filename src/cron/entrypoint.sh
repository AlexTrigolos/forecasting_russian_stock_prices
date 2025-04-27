#!/bin/sh

# Запуск Python-скрипта в фоновом режиме
/usr/src/app/venv/bin/python /usr/src/app/daytime.py >> /var/log/cron.log 2>&1 &

# # Экспорт переменных окружения
printenv | sed 's/^\(.*\)$/export \1/g' > /root/project_env.sh

# Запуск cron
cron -f -l 2
