import nltk
from nltk.chat.util import Chat, reflections
from nltk import word_tokenize, pos_tag

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Sample responses and patterns
pairs = [
    [
        r"my name is (.*)",
        ["Hello %1, how can I help you today?", ]
    ],
    [
        r"hi|hey|hello",
        ["Hello, how can I help you?", ]
    ],
    [
        r"what is your name?",
        ["My name is ChatBot.", ]
    ],
    [
        r"how are you?",
        ["I'm doing well, thank you! How can I assist you today?", ]
    ],
    [
        r"sorry (.*)",
        ["No problem at all!", ]
    ],
    [
        r"quit",
        ["Goodbye! Have a great day ahead.", ]
    ],
    [
        r"(.*)",
        ["I'm not sure how to respond to that. Could you please elaborate?", ]
    ],
]

# Reflections (optional)
reflections = {
    "i am": "you are",
    "i was": "you were",
    "i": "you",
    "i'd": "you would",
    "i've": "you have",
    "i'll": "you will",
    "my": "your",
    "you are": "I am",
    "you were": "I was",
    "you've": "I have",
    "you'll": "I will",
    "your": "my",
    "yours": "mine",
    "you": "me",
    "me": "you"
}

def extract_entities(text):
    words = word_tokenize(text)
    tagged = pos_tag(words)
    entities = [word for word, pos in tagged if pos == 'NNP']
    return entities

def analyze_sentiment(text):
    positive_words = ["good", "great", "awesome", "fantastic", "amazing"]
    negative_words = ["bad", "terrible", "horrible", "awful", "worst"]
    words = word_tokenize(text.lower())
    positive_score = len([word for word in words if word in positive_words])
    negative_score = len([word for word in words if word in negative_words])

    if positive_score > negative_score:
        return {"label": "POSITIVE", "score": positive_score}
    elif negative_score > positive_score:
        return {"label": "NEGATIVE", "score": negative_score}
    else:
        return {"label": "NEUTRAL", "score": 0}

def chatbot():
    print("Hi! I'm an advanced chatbot. Type 'quit' to exit.")
    chat = Chat(pairs, reflections)
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Goodbye! Have a great day!")
            break

        # Respond to user input
        response = chat.respond(user_input)

        # Named entity recognition
        entities = extract_entities(user_input)
        if entities:
            response += "\nI noticed the following entities: " + ", ".join(entities)

        # Sentiment analysis
        sentiment = analyze_sentiment(user_input)
        response += f"\nSentiment: {sentiment['label']} (Score: {sentiment['score']})"

        print("Bot:", response)

if __name__ == "__main__":
    chatbot()
