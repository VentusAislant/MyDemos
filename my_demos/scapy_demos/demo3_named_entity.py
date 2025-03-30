import spacy

# 命名实体识别
nlp = spacy.load("en_core_web_sm")
text = "Elon Musk founded SpaceX in 2002 in the United States."
doc = nlp(text)

for ent in doc.ents:
    print(f"{ent.text} - {ent.label_}")
