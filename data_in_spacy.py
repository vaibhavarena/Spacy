import spacy
from spacy.tokens import Doc, Span


# with open('news.txt', 'r', encoding='utf-8') as f:
#     data = f.read()

nlp = spacy.load('en_core_web_lg')

doc1 = nlp("I love coffee")

coffee_hash = nlp.vocab.strings["coffee"]
coffee_string = nlp.vocab.strings[coffee_hash]

print(coffee_hash)
print(coffee_string)

lexeme = nlp.vocab['coffee']    # context-independent vocab
print(lexeme.text, lexeme.orth, lexeme.is_alpha)

words = ['Hello', 'world', '!']
spaces = [True, False, False]

doc = Doc(nlp.vocab, words=words, spaces=spaces)

span = Span(doc, 0, 3)
span_with_label = Span(doc, 0, 3, label="GREETING")

doc.ents = [span_with_label]    # Add span to doc.ents

print([(ent.text, ent.label_) for ent in doc.ents])

#################################################
########### SEMANTIC SIMILARITY #################
#################################################

doc3 = nlp("I like Pizza")
doc4 = nlp("I like Pasta")

print(doc3.similarity(doc4))

span3 = Span(doc3, 2, 3)
span4 = Span(doc4, 2, 3)

print(span3, span4)
print(span3.similarity(span4))

# Picking pos_ combinations

sentence = nlp("Berlin looks like a nice city. Tom eats pizza")

sen_text = [token.text for token in sentence]
sen_pos = [token.pos_ for token in sentence]

for i, pos in enumerate(sen_pos):
    if pos == "PROPN":
        next_pos = sen_pos[i+1]
        if next_pos == "VERB":
            result = sen_text[i]
            print(result, sen_text[i+1])


# vectors in medium and large models
sen = nlp("Two bananas in pyjamas")

banana_vector = sen[1].vector
print(len(banana_vector))   # 300 length vector


