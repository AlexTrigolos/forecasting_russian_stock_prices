from sqlalchemy import Column, Integer, String
from database import Base, engine
from sqlalchemy import inspect


class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    description = Column(String)


tables_to_create = [Item]

existing_tables = inspect(engine).get_table_names()

for table in tables_to_create:
    if table.__tablename__ not in existing_tables:
        print(f"Таблица '{table.__tablename__}' не существует. Создание таблицы...")
        Base.metadata.create_all(engine)
    else:
        print(f"Таблица '{table.__tablename__}' уже существует.")
