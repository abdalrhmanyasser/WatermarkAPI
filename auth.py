from fastapi import Depends, HTTPException, status, Header
from fastapi.security import OAuth2PasswordRequestForm
from typing import Optional
from sqlalchemy.orm import Session
from database import get_db
from models import User, ApiKey
from datetime import datetime, timedelta
import secrets
import bcrypt

try:
    from jose import JWTError, jwt
except Exception:
    jwt = None

# Config
JWT_SECRET = secrets.token_urlsafe(32)
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_SECONDS = 3600


def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode('utf-8'), hashed.encode('utf-8'))
    except Exception:
        return False


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    if jwt is None:
        raise RuntimeError("python-jose is required for JWT support")
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(seconds=JWT_EXPIRE_SECONDS))
    to_encode.update({"exp": expire})
    encoded = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded


def get_current_user_from_jwt(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)) -> User:
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization header")
    token = parts[1]
    if jwt is None:
        raise HTTPException(status_code=500, detail="Server misconfigured: JWT library missing")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = int(payload.get("sub"))
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def verify_api_key(x_api_key: Optional[str] = Header(None), db: Session = Depends(get_db)) -> User:
    """Validate raw API key provided in X-API-Key header. Returns owning User."""
    if not x_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing X-API-Key header")
    # Expect format: <prefix>.<secret>
    parts = x_api_key.split('.', 1)
    if len(parts) != 2:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key format")
    prefix, secret = parts[0], parts[1]

    rec = db.query(ApiKey).filter(ApiKey.key_prefix == prefix).first()
    if not rec:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")

    # Verify the secret portion using bcrypt
    if not verify_password(secret, rec.hashed_secret):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")

    # update last_used timestamp
    rec.last_used = datetime.utcnow()
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec.user