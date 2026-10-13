"""Main server code"""

from urllib.parse import urlparse
import os
from datetime import datetime
import asyncio

import jwt
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
NEON_AUTH_URL = os.getenv("NEON_AUTH_URL")
_u = urlparse(NEON_AUTH_URL)
ORIGIN = f"{_u.scheme}://{_u.netloc}"

jwks_client = jwt.PyJWKClient(f"{NEON_AUTH_URL}/.well-known/jwks.json", cache_keys=True)
bearer = HTTPBearer()

app = FastAPI()
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

def get_current_user(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> dict:
    """Returns the current user or unauthorized"""
    token = creds.credentials
    try:
        key = jwks_client.get_signing_key_from_jwt(token).key
        return jwt.decode(
            token,
            key,
            algorithms=["EdDSA"],
            issuer=ORIGIN,
            audience=ORIGIN,
            options={"require": ["exp", "sub"]},
        )
    except jwt.PyJWTError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid token: {e}") from e


