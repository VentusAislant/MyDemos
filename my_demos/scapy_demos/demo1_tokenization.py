import spacy

# 分词
nlp = spacy.load("en_core_web_sm")
text = "The quick brown fox jumps over the lazy dog."
doc = nlp(text)

tokens = [token.text for token in doc]
print(tokens)
