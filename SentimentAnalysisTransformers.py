from transformers import pipeline
classifier = pipeline("sentiment-analysis", framework="pt")  # Force PyTorch

# Input sentence
sentence = "Despite the high price, the performance of the new MacBook is outstanding."

# Run sentiment analysis
result = classifier(sentence)[0]

# Print results
print(f"Sentiment: {result['label']}")
print(f"Confidence Score: {result['score']:.4f}")
