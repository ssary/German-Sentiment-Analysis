from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime

Base = declarative_base()

class Review(Base):
    __tablename__ = 'review'
    review_id = Column(Integer, primary_key=True)
    review = Column(String)
    timestamp = Column(DateTime)