from pydantic import BaseModel


class ItemCreate(BaseModel):
    name: str
    description: str


class Item(ItemCreate):
    id: int

    class Config:
        orm_mode = True
