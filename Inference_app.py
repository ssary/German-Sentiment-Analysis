import psycopg2
from datetime import datetime
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def database_connection():
    db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "german_sentiment",
    "user": "postgres",
    "password": "startengine1"
    }
    try:
        # Establishing the connection
        conn = psycopg2.connect(**db_config)

        # Creating a cursor object
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        record = cursor.fetchone()
        print("You are connected to - ", record, "\n")
        return conn, cursor
    except(Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)
        if conn:
            conn.close()

def process_input(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_classes = ['negative', 'neutral', 'positive']
    sentiment = sentiment_classes[predictions.argmax()]
    return sentiment

MODEL_PATH = 'E:\HTW\papers\Sentiment analysis\XLM-RoBERTa model'

#Load the pretrained model and it's tokenizer from file
xlm_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
xlm_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

'''
Doing it from the cloud instead of file

model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

'''

# connecting to the database
conn, cursor = database_connection()

print("Enter the reviews, one review on each line, enter empty line to get the sentiments and exit")
while True:
    user_text = input()
    if user_text == "":
        break

    sentiment = process_input(user_text, xlm_tokenizer, xlm_model)
    print(f'sentiment for {user_text} is {sentiment}')

    sentence_sentiment = input("Please specify the sentiment of the sentence:")
    actual_aspects = input("Please specify the aspects of the sentence seperated by commas:")
    actual_aspects = actual_aspects.split(',')
    
    aspect_sentiments = input("Please specify the sentiment of each aspect in the same order, seperated by commas:")
    aspect_sentiments = aspect_sentiments.split(',')

    timenow = datetime.now()

# Closing the cursor and connection
cursor.close()
conn.close()
