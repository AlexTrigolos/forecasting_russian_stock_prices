from loguru import logger
from fastapi import FastAPI, Depends, Request
from sqlalchemy.orm import Session
from schemas import Item, ItemCreate
from crud import create_item, get_items
from database import SessionLocal

f = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {name} | {level} | {message}"
# Настройка логгирования
logger.add(
    "logs/file_{time:YYYY-MM-DD}.log",
    format=f,
    rotation="1 day",
    retention="180 days",
    compression="zip",
    enqueue=True,
    backtrace=True,
    diagnose=True
)

app = FastAPI()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    client_ip = request.client.host
    request_method = request.method
    request_url = request.url.path

    if request.method == "POST":
        request_params = await request.json()
    else:
        request_params = request.query_params

    request = f'{request_method} {request_url}, параметры "{request_params}"'
    logger.info(f'Запрос от {client_ip}: {request}')

    response = await call_next(request)

    answer = f'{response.status_code} для {request_method} {request_url}'
    logger.info(f"Ответ статус: {answer}")

    return response


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/items/", response_model=Item)
def create_items(item: ItemCreate, db: Session = Depends(get_db)):
    return create_item(db=db, item=item)


@app.get("/items/", response_model=list[Item])
def read_items(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    items = get_items(db, skip=skip, limit=limit)
    return items
