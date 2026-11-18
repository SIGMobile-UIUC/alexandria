"""All BaseModel inherited classes"""

from typing import Literal

from pydantic import BaseModel, Field

class SentenceRequest(BaseModel):
  level: int

class CheckRequest(BaseModel):
  source: str
  answer: str
  source_lang: str

class AuthRequest(BaseModel):
  username: str = Field(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_]+$")
  password: str = Field(min_length=8, max_length=128)

class LoginRequest(BaseModel):
  username: str = Field(min_length=1, max_length=32)
  password: str = Field(min_length=1, max_length=128)

class XPRequest(BaseModel):
  amount: Literal[5, 10]
  reason: str = Field(default="translation", max_length=64)
