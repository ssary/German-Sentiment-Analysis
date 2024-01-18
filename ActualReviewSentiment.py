from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String

Base = declarative_base()

class ActualReviewSentiment(Base):
    __tablename__ = 'actual_review_sentiment'
    actual_sentiment_id = Column(Integer, primary_key=True)
    review_id = Column(Integer)
    sentiment = Column(String)