from sqlalchemy import Column, Integer, String
from database import Base, engine


class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    description = Column(String)


Base.metadata.create_all(bind=engine)
