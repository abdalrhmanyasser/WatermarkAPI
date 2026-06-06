from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, nullable=False, unique=True)
    api_key = Column(String, unique=True, nullable=False, index=True)

    secrets = relationship("WatermarkSecret", back_populates="user", cascade="all, delete-orphan")

class WatermarkSecret(Base):
    __tablename__ = "watermark_secrets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    algorithm = Column(String, nullable=False)
    payload = Column(String, nullable=False)  # JSON string of the numpy array

    user = relationship("User", back_populates="secrets")