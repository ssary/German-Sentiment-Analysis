import psycopg2
from datetime import datetime
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import MetaData
import os
from dotenv import load_dotenv
import requests


def load_settings():
    # Load the environment variables from the .env file
    load_dotenv('settings.env')

    # Reading the settings including the password
    settings = {
        "database_url": os.getenv("DATABASE_URL"),
        "api_key": os.getenv("API_KEY"),
        "model_path": os.getenv("MODEL_PATH")
    }

    return settings

Base = declarative_base()

class Review(Base):
    __tablename__ = 'review'
    review_id = Column(Integer, primary_key=True)
    review = Column(String)
    timestamp = Column(DateTime)

class PredictedReviewSentiment(Base):
    __tablename__ = 'predicted_review_sentiment'
    predicted_sentiment_id = Column(Integer, primary_key=True)
    review_id = Column(Integer)
    sentiment = Column(String)

class ActualReviewSentiment(Base):
    __tablename__ = 'actual_review_sentiment'
    actual_sentiment_id = Column(Integer, primary_key=True)
    review_id = Column(Integer)
    sentiment = Column(String)

class DatabaseManager:
    def __init__(self, db_url):
        """
        Initialize the database connection using the provided URL.

        Args:
        - db_url (str): A string containing the database URL.
        """
        self.engine = create_engine(db_url)
        self.meta = MetaData()
        self.meta.reflect(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_review(self, review_text, timestamp):
        """
        Add a review to the database.

        Args:
        - review_text (str): The text content of the review.
        - timestamp (datetime.datetime): The timestamp when the review was created or submitted.

        Returns:
        - int: The primary key ID of the newly added review record.
        """
        session = self.Session()
        new_review = Review(review=review_text, timestamp=timestamp)
        session.add(new_review)
        session.commit()
        review_id = new_review.review_id
        session.close()
        return review_id

    def add_predicted_sentiment(self, review_id, sentiment):
        """
        Add a predicted sentiment to the database.

        Args:
        - review_id (int): The ID of the review for which the sentiment is predicted.
        - sentiment (str): The predicted sentiment (e.g., 'positive', 'negative', 'neutral').

        Returns:
        None
        """
        session = self.Session()
        prediction = PredictedReviewSentiment(review_id=review_id, sentiment=sentiment)
        session.add(prediction)
        session.commit()
        session.close()

    def add_actual_sentiment(self, review_id, sentiment):
        """
        Add an actual sentiment to the database.

        Args:
        - review_id (int): The ID of the review for which the actual sentiment is provided.
        - sentiment (str): The actual sentiment (e.g., 'positive', 'negative', 'neutral').

        Returns:
        None
        """
        session = self.Session()
        actual_sentiment = ActualReviewSentiment(review_id=review_id, sentiment=sentiment)
        session.add(actual_sentiment)
        session.commit()
        session.close()

    def get_last_entries(self, table, limit=3):
        """
        Get the last 'n' entries from a specific table.

        Args:
        - table (sqlalchemy.ext.declarative.api.DeclarativeMeta): The SQLAlchemy table class representing the database table.
        - limit (int, optional): The number of entries to retrieve. Default is 3.

        Returns:
        - list: A list of ORM instances representing the last 'n' rows of the table.
        """
        session = self.Session()
        # Find the primary key column
        primary_key_column = table.__table__.primary_key.columns.values()[0]
        entries = session.query(table).order_by(primary_key_column.desc()).limit(limit).all()
        session.close()
        return entries
    
    def print_last_n_rows_all_tables(self, limit=3):
        """
        Prints the last 'n' rows of all tables in the database.

        Args:
        - limit (int, optional): The number of entries to retrieve from each table. Default is 3.

        Returns:
        None
        """
        session = self.Session()
        for table_name, table in self.meta.tables.items():
            print(f'\n{table_name} table:')
            primary_key_column = table.primary_key.columns.values()[0]
            last_rows = session.query(table).order_by(primary_key_column.desc()).limit(limit).all()
            for row in last_rows:
                print(row)
        session.close()


class SentimentModel:
    def __init__(self, model_path):
        """
        Initialize the sentiment model with the specified model path.

        Args:
        - model_path (str): The file path or model identifier for loading the model and tokenizer.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def predict_sentiment(self, text):
        """
        Predict the sentiment of the given text.

        Args:
        - text (str): The input text for sentiment prediction.

        Returns:
        - str: The predicted sentiment label ('negative', 'neutral', 'positive').
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():  # Ensures that no gradient calculation is done, which saves memory and computations
            outputs = self.model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment_classes = ['negative', 'neutral', 'positive']
        return sentiment_classes[predictions.argmax()]


settings = load_settings()
print(settings)
model_path = settings['model_path']
# uncomment to use model identifier that downloads the model and use it (if not downloaded on your machine or path is invalid)
# model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_model = SentimentModel(model_path)

db_url = settings['database_url']
db_manager = DatabaseManager(db_url)

print("Enter the reviews, one review on each line, enter empty line to get the sentiments and exit")
while True:
    user_text = input()
    if user_text == "":
        break
    
    # get the time right now to add it with the review
    timenow = datetime.now()

    # get each line as a review
    reviews = user_text.split('\n')

    # Iterate over the reviews one by one
    for review in reviews:
        predicted_sentiment = sentiment_model.predict_sentiment(user_text)
        print(f'sentiment for "{user_text}" is {predicted_sentiment}')

        actual_sentiment = input("Please specify the sentiment(positive, negative, neutral) of the sentence:").lower()

        actual_aspects = input("Please specify the aspects of the sentence seperated by commas:")
        actual_aspects = actual_aspects.split(',')
        
        aspect_sentiments = input("Please specify the sentiment(positive, negative, neutral) of each aspect in the same order, seperated by commas:")
        aspect_sentiments = aspect_sentiments.split(',')

        # Insert review to the review table and return the id
        review_id = db_manager.add_review(review, datetime.now())

        # Insert the prediction alongwith the review_id
        db_manager.add_predicted_sentiment(review_id, predicted_sentiment)

        # Insert actual sentiment if the user entered it correctly
        if actual_sentiment in ['positive', 'negative', 'neutral']:
            db_manager.add_actual_sentiment(review_id, actual_sentiment)

        #Fetching last three rows of each table in the database
        db_manager.print_last_n_rows_all_tables()

# Closing the cursor and connection
print('Ciao !')
