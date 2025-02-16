# Используем официальный образ Nginx на базе Alpine
FROM nginx:1.27-alpine

# Устанавливаем рабочую директорию
WORKDIR /etc/nginx

# Копируем конфигурационный файл Nginx
COPY ./nginx.conf ./nginx.conf

# Копируем статические файлы
COPY ./static /var/www/html/static

# Копируем SSL-сертификаты
COPY ./certificates/certificate_russian_stocks.crt /etc/ssl/certs/certificate_russian_stocks.crt
COPY ./certificates/certificate.key /etc/ssl/private/certificate.key
# COPY ./certificates/nginx-selfsigned.crt /etc/ssl/certs/certificate_russian_stocks.crt
# COPY ./certificates/nginx-selfsigned.key /etc/ssl/private/certificate.key

RUN chmod 0600 /etc/ssl/private/certificate.key

# Открываем порты 80 и 443
EXPOSE 80
EXPOSE 443

# Указываем команду для запуска Nginx
CMD ["nginx", "-g", "daemon off;"]
