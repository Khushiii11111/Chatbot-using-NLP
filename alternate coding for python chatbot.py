import nltk
import spacy
from transformers import pipeline # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load HuggingFace transformers pipeline for sentiment analysis
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", revision="714eb0f")

# Initialize TF-IDF Vectorizer and Logistic Regression for intent recognition
vectorizer = TfidfVectorizer()
classifier = LogisticRegression()

# Training data for intent recognition
training_data = [
    ("Hi", "greeting"),
    ("Hello", "greeting"),
    ("Hey", "greeting"),
    ("Bye", "goodbye"),
    ("See you", "goodbye"),
    ("Goodbye", "goodbye"),
    ("What is your name?", "name_query"),
    ("Who are you?", "name_query"),
    ("How are you?", "how_are_you"),
    ("What can you do?", "capabilities_query"),
]

X_train = [text for text, label in training_data]
y_train = [label for text, label in training_data]

X_train_tfidf = vectorizer.fit_transform(X_train)
classifier.fit(X_train_tfidf, y_train)

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def analyze_sentiment(text):
    return sentiment_analyzer(text)[0]

def respond_greeting():
    return "Hello! How can I assist you today?"

def respond_goodbye():
    return "Goodbye! Have a great day!"

def respond_name_query():
    return "I am an advanced chatbot created to assist you."

def respond_how_are_you():
    return "I'm just a bunch of code, but I'm here to help you!"

def respond_capabilities_query():
    return "I can chat with you, answer questions, and provide information."

def chatbot():
    print("Hi! I'm an advanced chatbot. Type 'quit' to exit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print(respond_goodbye())
            break
        
        user_input_tfidf = vectorizer.transform([user_input])
        intent = classifier.predict(user_input_tfidf)[0]
        
        if intent == "greeting":
            response = respond_greeting()
        elif intent == "goodbye":
            response = respond_goodbye()
        elif intent == "name_query":
            response = respond_name_query()
        elif intent == "how_are_you":
            response = respond_how_are_you()
        elif intent == "capabilities_query":
            response = respond_capabilities_query()
        else:
            response = "I'm not sure how to respond to that. Could you please elaborate?"
        
        entities = extract_entities(user_input)
        if entities:
            response += "\nI noticed the following entities: " + ", ".join([f"{ent} ({label})" for ent, label in entities])
        
        sentiment = analyze_sentiment(user_input)
        response += f"\nSentiment: {sentiment['label']} (Confidence: {sentiment['score']:.2f})"
        
        print("Bot:", response)

if __name__ == "__main__":
    chatbot()