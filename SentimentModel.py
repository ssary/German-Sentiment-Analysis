from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class SentimentModel:
    def __init__(self, model_path):
        """
        Initialize the sentiment model with the specified model path.

        Args:
        - model_path (str): The file path or model identifier for loading the model and tokenizer.
        """
        #model_path = r'C:\Users\saryn\Downloads\Shared folder\fine tune xlmroberta SA\Results HPC 100K\results\best_model'
        #self.model = AutoModelForSequenceClassification.from_pretrained(model_path, from_tf=True)
        #self.tokenizer = AutoTokenizer.from_pretrained(model_path, from_tf=True)
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

