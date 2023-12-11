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

def insert_review(conn, cursor, review, timestamp):
    insertion_query = 'INSERT INTO review(review, timestamp) VALUES(%s, %s) RETURNING review_id'
    cursor.execute(insertion_query, (review, timestamp))
    conn.commit()
    return cursor.fetchone()[0]

def insert_sentiment_prediction(conn, cursor, review_id, sentiment):
    query = 'INSERT INTO predicted_review_sentiment(review_id, sentiment) VALUES(%s, %s)'
    cursor.execute(query, (review_id, sentiment))
    conn.commit()

def insert_actual_sentiment_prediction(conn, cursor, review_id, sentiment):
    query = 'INSERT INTO actual_review_sentiment(review_id, sentiment) VALUES(%s, %s)'
    cursor.execute(query, (review_id, sentiment))
    conn.commit()

def view_table(cursor, table_name, id_name):
    # print the last 3 rows in the table
    query = f'SELECT * from {table_name} ORDER BY {id_name} DESC LIMIT 3'
    cursor.execute(query)
    row = cursor.fetchone()
    while row is not None:
        print(row)
        row = cursor.fetchone()


# def row_insertion(review, predicted_review_sentiment, actual_review_sentiment, aspects, actual_aspects, )
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
    
    # get the time right now to add it with the review
    timenow = datetime.now()

    # get each line as a review
    reviews = user_text.split('\n')

    # Iterate over the reviews one by one
    for review in reviews:
        predicted_sentiment = process_input(user_text, xlm_tokenizer, xlm_model)
        print(f'sentiment for "{user_text}" is {predicted_sentiment}')

        actual_sentiment = input("Please specify the sentiment(positive, negative, neutral) of the sentence:").lower()

        actual_aspects = input("Please specify the aspects of the sentence seperated by commas:")
        actual_aspects = actual_aspects.split(',')
        
        aspect_sentiments = input("Please specify the sentiment(positive, negative, neutral) of each aspect in the same order, seperated by commas:")
        aspect_sentiments = aspect_sentiments.split(',')

        # Insert review to the review table and return the id
        review_id = insert_review(conn, cursor, review, timenow)

        # Insert the prediction alongwith the review_id
        insert_sentiment_prediction(conn, cursor, review_id, predicted_sentiment)
        
        # Insert actual sentiment if the user entered it correctly
        if actual_sentiment in ['positive', 'negative', 'neutral']:
            insert_actual_sentiment_prediction(conn, cursor, review_id, actual_sentiment)
        
        print('\nreviews table')
        view_table(cursor, 'review', 'review_id')
        print('\npredicted reviews sentiment table')
        view_table(cursor, 'predicted_review_sentiment', 'predicted_sentiment_id')
        print('\nactual_review_sentiment table')
        view_table(cursor, 'actual_review_sentiment', 'actual_sentiment_id')

# Closing the cursor and connection
print('Ciao !')
cursor.close()
conn.close()
