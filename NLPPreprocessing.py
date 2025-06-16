import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data (only neded once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

def nlp_preprocessing(sentence):
    # 1. Tokenize
    tokens = word_tokenize(sentence)
    print("Original Tokens:", tokens)

    # 2. Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens_no_stopwords = [word for word in tokens if word.lower() not in stop_words]
    print("Tokens Without Stopwords:", tokens_no_stopwords)

    # 3. Apply stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in tokens_no_stopwords]
    print("Stemmed Words:", stemmed_words)

# Test sentence
sentence = "NLP techniques are used in virtual assistants like Alexa and Siri."
nlp_preprocessing(sentence)
