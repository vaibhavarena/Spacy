import spacy

nlp = spacy.load('en_core_web_md')

text = 'A paragraph is a series of related sentences developing a central idea, called the topic. Try to think about paragraphs in terms of thematic unity: a paragraph is a sentence or a group of sentences that supports one central, unified idea. Paragraphs add one idea at a time to your broader argument.'


# nlp.add_pipe(component, first=True)           Adding components to your pipeline
# nlp.add_pipe(component, last=True)            Default behaviour
# nlp.add_pipe(component, before="ner")
# nlp.add_pipe(component, after="tagger")

def len_component(doc):
    print('Doc length :', len(doc))
    return doc

nlp.add_pipe(len_component, first=True)
print("Pipeline :", nlp.pipe_names)

doc = nlp(text)

print(nlp.pipe_names)
print(nlp.pipeline)

for i in doc.noun_chunks:
    print(i)