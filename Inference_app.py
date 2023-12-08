import psycopg2
from datetime import datetime
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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