# forecasting_russian_stock_prices
Выпускная квалификационная работа по прогнозированию цен российских акций

# Запуск кода локально
Для телеграм бота необходимо в src/tg_bot/.env установить BOT_TOKEN (токен вашего бота в тг)
Для fastapi и docker-compose прописать в src/fastapi/app/.env и src/.env POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD (название бд, имя пользователя с доступом и его пароль)

![пример структуры с .env файлами](images/image.png)

`cd src`

`docker-compose up --build -d`

Будут запущены fastapi, stramlit, postgres и telegram_bot

# Удаление контейнеров
`docker-compose down` с опцией -v будут удалены volums с бд

# Пример работы
Пример работы docker-compose и сервисов можно посмотреть здесь
https://github.com/AlexTrigolos/forecasting_russian_stock_prices/blob/master/records/record_rus_stocks.mp4
