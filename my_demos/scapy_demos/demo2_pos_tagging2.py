import spacy

# 加载模型
nlp = spacy.load("en_core_web_md")

# 解析文本
text = "The quick brown fox jumps over the lazy dog and persons quickly."
doc = nlp(text)

# 名词
nouns = [token.text for token in doc if token.pos_ == "NOUN"]
# 动词
verbs = [token.text for token in doc if token.pos_ == "VERB"]
# 形容词
adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
# 副词
adverbs = [token.text for token in doc if token.pos_ == "ADV"]
# 代词
pronouns = [token.text for token in doc if token.pos_ == "PRON"]
# 冠词
articles = [token.text for token in doc if token.pos_ == "DET"]
# 介词
prepositions = [token.text for token in doc if token.pos_ == "ADP"]
# 数词
numbers = [token.text for token in doc if token.pos_ == "NUM"]
# 连词
conjunctions = [token.text for token in doc if token.pos_ == "CCONJ"]
# 从属连词 although， because， if
subordinating_conjunctions = [token.text for token in doc if token.pos_ == "SCONJ"]
# 感叹词
interjections = [token.text for token in doc if token.pos_ == "INTJ"]
# 辅助动词 is, have, will。
auxiliary_verbs = [token.text for token in doc if token.pos_ == "AUX"]
# 符号
symbols = [token.text for token in doc if token.pos_ == "SYM"]
# 空格
spaces = [token.text for token in doc if token.pos_ == "SPACE"]
# 其他
other_tokens = [token.text for token in doc if token.pos_ == "X"]

print("Nouns:", nouns)
print("Verbs:", verbs)
print("Adjectives:", adjectives)
print("Adverbs:", adverbs)
print("Pronouns:", pronouns)
print("Articles:", articles)
print("Prepositions:", prepositions)
print("Numbers:", numbers)
print("Conjunctions:", conjunctions)
print("Subordinating Conjunctions:", subordinating_conjunctions)
print("Interjections:", interjections)
print("Auxiliary Verbs:", auxiliary_verbs)
print("Symbols:", symbols)
print("Spaces:", spaces)
print("Other Tokens:", other_tokens)