import os
from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

hasher = PasswordHasher()
DUMMY_HASH = hasher.hash("dummy-password")


def hash_password(password: str) -> str:
    return hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return hasher.verify(password_hash, password)
    except (VerifyMismatchError, InvalidHashError):
        return False


def create_token(user_id: int) -> str:
    payload = {
        "sub": str(user_id),
        "exp": datetime.now(timezone.utc) + timedelta(days=30),
    }
    return jwt.encode(payload, os.environ["JWT_SECRET"], algorithm="HS256")


def decode_token(token: str) -> int:
    payload = jwt.decode(token, os.environ["JWT_SECRET"], algorithms=["HS256"])
    return int(payload["sub"])
