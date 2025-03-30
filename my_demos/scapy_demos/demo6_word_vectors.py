import spacy

nlp = spacy.load("en_core_web_md")  # 需要使用 medium 或 large 版模型
word1 = nlp("astronaut")
word2 = nlp("spacecraft")
word3 = nlp("apple")

print(f"Similarity between '{word1.text}' and '{word2.text}': {word1.similarity(word2)}")
print(f"Similarity between '{word1.text}' and '{word3.text}': {word1.similarity(word3)}")
