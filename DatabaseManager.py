from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import MetaData
from Review import Review
from PredictedReviewSentiment import PredictedReviewSentiment
from ActualReviewSentiment import ActualReviewSentiment

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
        Add a predicted sentiment of specific review to the database.

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
        Add an actual sentiment to specific review to the database.

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
