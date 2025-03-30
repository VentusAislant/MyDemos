import spacy

nlp = spacy.load("en_core_web_sm")
text = "The astronaut piloted the spacecraft to the moon."
doc = nlp(text)

for token in doc:
    print(f"{token.text} <--({token.dep_})-- {token.head.text}")
