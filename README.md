# forecasting_russian_stock_prices
Выпускная квалификационная работа по прогнозированию цен российских акций

# Самое актуальное
Для запуска и остановки использовать `--env-file .dev.env -f docker-compose.dev.yml` для локалки, например
`docker-compose --env-file .dev.env -f docker-compose.dev.yml up --build -d` для локального билда и запуска.

Теперь один `.env` файл внутри `src` папки, он хранит и передает все секреты там, где это нужно.

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

[![Watch the video](records/record_rus_stocks.mp4)](records/record_rus_stocks.mp4)
