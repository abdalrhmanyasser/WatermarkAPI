from fastapi import Depends, HTTPException, status, Header
from typing import Optional
from database import get_db
from datetime import datetime, timedelta
import secrets
import bcrypt
from bson import ObjectId

try:
    from jose import JWTError, jwt
except Exception:
    jwt = None

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

def get_current_user_from_jwt(authorization: Optional[str] = Header(None), db = Depends(get_db)):
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization header")
    token = parts[1]
    
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        
    user = db.users.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user

def verify_api_key(x_api_key: Optional[str] = Header(None), db = Depends(get_db)):
    if not x_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing X-API-Key header")
        
    parts = x_api_key.split('.', 1)
    if len(parts) != 2:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key format")
        
    prefix, secret = parts[0], parts[1]

    rec = db.api_keys.find_one({"key_prefix": prefix})
    if not rec or not verify_password(secret, rec["hashed_secret"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")

    db.api_keys.update_one({"_id": rec["_id"]}, {"$set": {"last_used": datetime.utcnow()}})
    
    user = db.users.find_one({"_id": rec["user_id"]})
    return user