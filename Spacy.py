import spacy

# Load the small English model
nlp = spacy.load("en_core_web_sm")

# Input sentnce
sentence = "Barack Obama served as the 44th President of the United States and won the Nobel Peace Prize in 2009."

# Process the sentence
doc = nlp(sentence)

# Extract and print named entities
for ent in doc.ents:
    print(f"Text: {ent.text}\nLabel: {ent.label_}\nStart: {ent.start_char}, End: {ent.end_char}\n")
