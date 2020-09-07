from spacy.lang.en import English
from spacy.matcher import Matcher
import spacy

# nlp = English()
nlp = spacy.load("en_core_web_sm")  #python -m spacy download en_core_web_sm

doc = nlp("Agent Smith Neo at In 1990, more than 60% of people in East Asia were in extreme poverty. "
    "Now less than 4% are.")

doc1 = nlp("Upcoming iPhone X date revealed")
doc2 = nlp("I have bought a smartphone. Now I'm buying apps.")

print(spacy.explain('PROPN'))

for token in doc:
    print(token.i, token.text, token.pos_, token.dep_, token.head.text)

for ent in doc.ents:
    print(ent.text, ent.label_)

matcher = Matcher(nlp.vocab)
pattern = [{"TEXT" : "iPhone"}, {"TEXT" : "X"}]
matcher.add("IPHONE_PATTERN", None, pattern)

matches = matcher(doc1)

for match_id, start, end in matches:
    print(doc1[start:end].text)


matcher_c = Matcher(nlp.vocab)
pattern = [{"LEMMA":"buy"},
           {"POS":"DET", "OP":"?"},    #OP operator    ?  0 or 1      +  1 or more      *  0 or more      !  0 
           {"POS":"NOUN"}]
matcher_c.add("BUYING", None, pattern)

matches_c = matcher_c(doc2)

for match_id, start, end in matches_c:
    print(doc2[start:end].text)



nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

doc = nlp(
    "Features of the app include a beautiful design, smart search, automatic "
    "labels and optional voice responses."
)

# Write a pattern for adjective plus one or two nouns
pattern = [{"POS": "ADJ"}, {"POS": "NOUN"}, {"POS": "NOUN", "OP": "?"}]

# Add the pattern to the matcher and apply the matcher to the doc
matcher.add("ADJ_NOUN_PATTERN", None, pattern)
matches = matcher(doc)
print("Total matches found:", len(matches))

# Iterate over the matches and print the span text
for match_id, start, end in matches:
    print("Match found:", doc[start:end].text)