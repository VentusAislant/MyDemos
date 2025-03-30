import spacy

# 词性标注
nlp = spacy.load("en_core_web_sm")
text = "The quick brown fox jumps over the lazy dog."
doc = nlp(text)

for token in doc:
    print(f"{token.text} <==> {token.pos_}")
