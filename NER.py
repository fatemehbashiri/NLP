import spacy
from spacy import displacy
from spacy import tokenizer

#you need to download en_core_web_sm first using this cmline :
# "python -m spacy download en_core_web_sm" in terminal
nlp = spacy.load('en_core_web_sm')

# Load the text and process it
# I copied the text from python wiki
text = ("The Little Prince is an honest and beautiful story about loneliness,"
        " friendship, sadness, and love. The prince is a small boy from a tiny planet "
        "(an asteroid to be precise), who travels the universe,"
        " planet-to-planet, seeking wisdom. On his journey, he discovers the unpredictable nature of adults.")

text_fa = ("سلام نام من فاطمه است و در حوزه داده کاوی کار میکنم ، شما چطور؟")
# text2 = # copy the paragraphs from  https://www.python.org/doc/essays/
doc = nlp(text)

sentences = list(doc.sents)
print(sentences)

# tokenization
for token in doc:
    print(token.text)

# print entities
ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]
print(ents)
# now we use displaycy function on doc2
displacy.render(doc, style='ent', jupyter=True)

# if doc.ents() is an empty list you should check "en_core_web_sm" using this line :
# python -m spacy validate
# or this line
# python -m spacy inform en_core_web_sm