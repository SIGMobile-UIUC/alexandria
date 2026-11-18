"""Main server code"""

from urllib.parse import urlparse
import os
from datetime import datetime
import asyncio

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, status, Request, Query
from contextlib import asynccontextmanager
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from typing import Annotated
from fastapi.middleware.cors import CORSMiddleware
from alexandria_models.inference import AlexandriaModels
from classes import *
import jwt
from auth import DUMMY_HASH, create_token, decode_token, hash_password, verify_password
from psycopg.errors import UniqueViolation
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads in model and passes as state"""
    model = AlexandriaModels.load("./artifacts")

    load_dotenv()

    pool = None
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        pool = AsyncConnectionPool(database_url, kwargs={"row_factory": dict_row}, open=False)
        await pool.open()

    yield {
        "model": model,
        "pool": pool
    }

    if pool is not None:
        await pool.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    """Returns status 200 if alive"""
    return {"status": 200}

bearer = HTTPBearer()

def current_user_id(credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer)]) -> int:
    try:
        return decode_token(credentials.credentials)
    except (jwt.PyJWTError, KeyError, ValueError):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")

def get_pool(request: Request):
    pool = request.state.pool
    if pool is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database not configured")
    return pool

@app.post("/register", status_code=status.HTTP_201_CREATED)
async def register(req: AuthRequest, request: Request):
    pool = get_pool(request)
    password_hash = await asyncio.to_thread(hash_password, req.password)
    async with pool.connection() as conn:
        try:
            cur = await conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id, username, xp",
                (req.username, password_hash),
            )
        except UniqueViolation:
            raise HTTPException(status.HTTP_409_CONFLICT, "Username already taken")
        user = await cur.fetchone()
    return {"token": create_token(user["id"]), "user": user}

@app.post("/login")
async def login(req: LoginRequest, request: Request):
    pool = get_pool(request)
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT id, username, xp, password_hash FROM users WHERE username = %s",
            (req.username,),
        )
        user = await cur.fetchone()
    password_hash = user["password_hash"] if user else DUMMY_HASH
    valid = await asyncio.to_thread(verify_password, password_hash, req.password)
    if user is None or not valid:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid username or password")
    user.pop("password_hash")
    return {"token": create_token(user["id"]), "user": user}

@app.get("/sentence")
async def sentence(req: Annotated[SentenceRequest, Query()], request: Request):
    return request.state.model.sentences(level=req.level, n=1, mode="bank")

@app.get("/check")
async def check(req: Annotated[CheckRequest, Query()], request: Request):
    return request.state.model.evaluate(source=req.source, answer=req.answer, source_lang=req.source_lang)

@app.post("/xp")
async def award_xp(req: XPRequest, request: Request, user_id: Annotated[int, Depends(current_user_id)]):
    pool = get_pool(request)
    async with pool.connection() as conn:
        cur = await conn.execute(
            "UPDATE users SET xp = xp + %s, updated_at = now() WHERE id = %s RETURNING xp",
            (req.amount, user_id),
        )
        row = await cur.fetchone()
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
        await conn.execute(
            "INSERT INTO xp_events (user_id, amount, reason) VALUES (%s, %s, %s)",
            (user_id, req.amount, req.reason),
        )
    return {"xp": row["xp"]}
