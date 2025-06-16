# Home_assignment3
# CS5720: Neural Networks and Deep Learning  
### Home Assignment 3 – Summer 2025  
**University of Central Missouri**  
**Student Name:** Sharath Chandra Seriyala
**Student ID:** 700776646

--------------------------------------------------------------------------------------------

## 🔍 Assignment Overview

This assignment covers various key topics in NLP and deep learning including:

- Character-level text generation using LSTM
- Basic NLP preprocessing (tokenization, stopword removal, stemming)
- Named Entity Recognition (NER) using SpaCy
- Scaled Dot-Product Attention (as used in Transformers)
- Sentiment analysis using HuggingFace Transformers

-----------------------------------------------------------------------------------------------------

###  Q1: RNN for Text Generation
- Trained an LSTM model on a character-level dataset (`shakespeare.txt`).
- Implemented text generation using temperature sampling.
- Explained the effect of temperature on randomness.
5. What does temperature do?
    A lower temperature (e.g., 0.2) makes the output more deterministic (less random).
    A higher temperature (e.g., 1.0) makes the model take more creative or risky guesses.



----------------------------------------------------------------------------------------------------

### Q2: NLP Preprocessing Pipeline
- Tokenized input sentence
- Removed common English stopwords using NLTK
- Applied stemming using Porter Stemmer
- ### Code Output:
**Original Tokens:** `['NLP', 'techniques', 'are', 'used', 'in', 'virtual', 'assistants', 'like', 'Alexa', 'and', 'Siri', '.']`  
**Without Stopwords:** `['NLP', 'techniques', 'used', 'virtual', 'assistants', 'like', 'Alexa', 'Siri']`   **After Stemming:** `['nlp', 'techniqu', 'use', 'virtual', 'assist', 'like', 'alexa', 'siri']`

5. Difference between stemming and lemmatization:
Stemming chops off word endings roughly to get the root (may not be a real word).
Example: "running" → "run" or sometimes "runn"
Lemmatization uses vocabulary and morphological analysis to get the correct base form (lemma).
Example: "running" → "run"

- Why remove stop words? When might it be harmful?
Useful:
Removing stop words reduces noise and dimensionality, improving efficiency and focusing on important content words in tasks like topic modeling or information retrieval.
Harmful:
Sometimes stop words carry meaning (e.g., negations "not", "no") or are essential for context in sentiment analysis or language generation. Removing them could degrade performance.

-----------------------------------------------------------------------------------------------
### Q3: Spacy
- Text, Label, Start, End details are printed
1. How does NER differ from POS tagging in NLP?
NER (Named Entity Recognition) identifies real-world entities (e.g., people, organizations, dates) in a text.
POS (Part-of-Speech) tagging assigns grammatical roles (e.g., noun, verb, adjective) to each word.
Example with "Barack Obama":
NER: 'Barack Obama' → PERSON
POS: 'Barack' → NNP (proper noun), 'Obama' → NNP (proper noun)
So, NER is semantic (what the word refers to), while POS tagging is syntactic (how the word functions in a sentence).

2. Two applications that use NER in the real world:
Financial News Monitoring:
Extracts company names, stock symbols, dates, and monetary values from articles.
Example: Bloomberg or Reuters auto-tagging "Apple Inc." or "$5 billion" for market trend analysis.
Search Engines (e.g., Google, Bing):
Improves query understanding by detecting names, places, or dates.
Example: A search for "movies by Christopher Nolan" uses NER to recognize "Christopher Nolan" as a PERSON.

------------------------------------------------------------------------------------------------------
### Q4: Scaled Dot-Product Attention 
- Attention weights and Final output is printed

1. Why do we divide the attention score by √d in the scaled dot-product attention formula?
We divide by √d (where d is the dimension of the key vectors) to:
Prevent large dot products from causing extremely small gradients when passed through the softmax function.
Stabilize training, especially when d is large, as dot products can grow significantly in magnitude, leading to sharper distributions after softmax (which can hinder learning).
Example: Without scaling, dot products like Q·Kᵀ could grow large (e.g., >10), making softmax outputs overly confident (close to 0 or 1), which hurts learning.

2. How does self-attention help the model understand relationships between words in a sentence?
Self-attention allows the model to:
Weigh the importance of other words in a sentence when processing each word.
Capture context and dependencies, even if words are far apart in the sequence.
Example: In “The cat that chased the mouse ran,” self-attention helps associate “cat” with “ran” despite “mouse” being in between, enabling better understanding of the sentence's structure and meaning.

------------------------------------------------------------------------------------------------------
### Q5: Sentiment Analysis using HuggingFace Transformers
- Printed Sentiment and Confidence Score
1. What is the main architectural difference between BERT and GPT? Which uses an encoder and which uses a decoder?
BERT (Bidirectional Encoder Representations from Transformers):
Uses only the encoder part of the Transformer architecture.
It processes input bidirectionally, meaning it looks at both the left and right context of a word.
GPT (Generative Pre-trained Transformer):
Uses only the decoder part of the Transformer.
It is unidirectional (left-to-right), focusing on generating text one token at a time.
Summary:
BERT → Encoder → Bidirectional → Good for understanding.
GPT → Decoder → Unidirectional → Good for text generation.

2. Why is using pre-trained models (like BERT or GPT) beneficial for NLP applications instead of training from scratch?
Saves Time and Resources: Training models from scratch requires massive datasets and high computational power (often GPUs/TPUs for days or weeks).
Pre-trained on Large Corpora: Models like BERT and GPT are already trained on billions of words, capturing general language patterns, syntax, and semantics.
Better Accuracy: Fine-tuning pre-trained models on specific tasks (like sentiment analysis, NER, QA) usually results in higher performance than models trained from scratch.
Easier Deployment: HuggingFace provides user-friendly APIs for instant deployment without deep knowledge of the architecture.
In short, pre-trained models provide a powerful starting point and reduce the barrier to building advanced NLP applications.


-----------------------------------------------------------------------------------------------------------



