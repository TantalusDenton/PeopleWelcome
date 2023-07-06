from typing import Optional, List
from uuid import UUID, uuid4
from pydantic import BaseModel
from enum import Enum

class Image(BaseModel):
    image_id: int
    label: str
    #score: str

class Character(BaseModel):
    User: str
    Ai: str