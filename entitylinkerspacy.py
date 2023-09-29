from spacy.pipeline.entity_linker import DEFAULT_NEL_MODEL

# Construction from class
from spacy.pipeline import EntityLinker


config = {
    "labels_discard": [],
    "n_sents": 0,
    "incl_prior": True,
    "incl_context": True,
    "model": DEFAULT_NEL_MODEL,
    "entity_vector_length": 64,
    "get_candidates": {"@misc": "spacy.CandidateGenerator.v1"},
    "threshold": None,
}

# Construction via add_pipe with default model
entity_linker = nlp.add_pipe("entity_linker", config=config)

# Construction via add_pipe with custom model
config = {"model": {"@architectures": "my_el.v1"}}
entity_linker = nlp.add_pipe("entity_linker", config=config)


entity_linker = EntityLinker(nlp.vocab, model)


doc = nlp("This is a sentence.")
entity_linker = nlp.add_pipe("entity_linker")
# This usually happens under the hood
processed = entity_linker(doc)

entity_linker = nlp.add_pipe("entity_linker")
for doc in entity_linker.pipe(docs, batch_size=50):
    pass


def create_kb(vocab):
    kb = InMemoryLookupKB(vocab, entity_vector_length=128)
    kb.add_entity(...)
    kb.add_alias(...)
    return kb


entity_linker = nlp.add_pipe("entity_linker")
entity_linker.initialize(lambda: examples, nlp=nlp, kb_loader=my_kb)


entity_linker = nlp.add_pipe("entity_linker")
kb_ids = entity_linker.predict([doc1, doc2])

entity_linker = nlp.add_pipe("entity_linker")
entity_linker.set_kb(create_kb)

entity_linker = nlp.add_pipe("entity_linker")
kb_ids = entity_linker.predict([doc1, doc2])
entity_linker.set_annotations([doc1, doc2], kb_ids)

entity_linker = nlp.add_pipe("entity_linker")
optimizer = nlp.initialize()
losses = entity_linker.update(examples, sgd=optimizer)
