from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String

Base = declarative_base()
class PredictedReviewSentiment(Base):
    __tablename__ = 'predicted_review_sentiment'
    predicted_sentiment_id = Column(Integer, primary_key=True)
    review_id = Column(Integer)
    sentiment = Column(String)