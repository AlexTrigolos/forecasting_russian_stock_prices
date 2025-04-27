DEV_COMPOSE_FILE=docker-compose.dev.yml
PROD_COMPOSE_FILE=docker-compose.prod.yml
DEV_ENV=.dev.env
PROD_ENV=.prod.env

# Сборка и запуск контейнера
start: build up

# Команда запускает контейнеры в фоновом режиме
up:
	docker-compose --env-file $(DEV_ENV) -f $(DEV_COMPOSE_FILE) up -d

# Останавливает и удаляет контейнеры, сети и образы
down:
	docker-compose --env-file $(DEV_ENV) -f $(DEV_COMPOSE_FILE) down -v

# Перезапускает все запущенные контейнеры
restart:
	docker-compose --env-file $(DEV_ENV) -f $(DEV_COMPOSE_FILE) restart

# Позволяет просматривать логи контейнеров в реальном времени
logs:
	docker-compose --env-file $(DEV_ENV) -f $(DEV_COMPOSE_FILE) logs -f

# Отсанавливаем контейнеры
stop:
	docker-compose --env-file $(DEV_ENV) -f $(DEV_COMPOSE_FILE) stop

# Удаляет все остановленные контейнеры, освобождая ресурсы
clean:
	docker-compose --env-file $(DEV_ENV) -f $(DEV_COMPOSE_FILE) rm -f

# Сборка образов
build:
	docker-compose --env-file $(DEV_ENV) -f $(DEV_COMPOSE_FILE) build

# Позволяет выполнять команды внутри запущенного контейнера
exec:
	docker-compose --env-file $(DEV_ENV) -f $(DEV_COMPOSE_FILE) exec $(SERVICE) $(COMMAND)

.PHONY: up down restart logs stop clean build exec
