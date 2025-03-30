import spacy

nlp = spacy.load("en_core_web_sm")
text = "NASA launched a new mission. The spacecraft is heading to Mars."
doc = nlp(text)

for sent in doc.sents:
    print(sent.text)
