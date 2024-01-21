from datetime import datetime
import os
from dotenv import load_dotenv
from SentimentModel import SentimentModel
from DatabaseManager import DatabaseManager

def load_settings():
    # Load the environment variables from the .env file
    load_dotenv('settings.env')

    # Reading the settings including the password
    settings = {
        "database_url": os.getenv("DATABASE_URL"),
        "api_key": os.getenv("API_KEY"),
        "model_path": os.getenv("MODEL_PATH"),
        "tokenizer_path": os.getenv("TOKENIZER_PATH")
    }

    return settings

settings = load_settings()
model_path = settings['model_path']
tokenizer_path = settings['tokenizer_path']
print(model_path)
# uncomment to use model identifier that downloads the model and use it (if not downloaded on your machine or path is invalid)
# model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
# tokenizer_path = model_path
sentiment_model = SentimentModel(model_path, tokenizer_path)

db_url = settings['database_url']
db_manager = DatabaseManager(db_url)
review_text = ""


def handle_on_click(review):
    predicted_sentiment = sentiment_model.predict_sentiment(review)
    print(f'sentiment for "{review}" is {predicted_sentiment}')
    review_id = db_manager.add_review(review, datetime.now())
    db_manager.add_predicted_sentiment(review_id, predicted_sentiment)
    db_manager.print_last_n_rows_all_tables()
    return predicted_sentiment



from View.gui import main_loop
main_loop()