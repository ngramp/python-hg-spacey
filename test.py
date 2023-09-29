import spacy
from spacy.pipeline.entity_linker import EntityLinker
from spacy.pipeline.entity_linker import DEFAULT_NEL_MODEL

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
# Load the SpaCy model with the EntityLinker component
nlp = spacy.load("en_core_web_sm")

# Construction via add_pipe with default model
nlp.add_pipe("entity_linker", config=config)


# Create an EntityLinker instance and add it to the pipeline
# entity_linker = EntityLinker()
# nlp.add_pipe("entity_linker")

# Process text
text = "Apple Inc. was founded by Steve Jobs."
doc = nlp(text)

# Iterate through entities and print their linked information
for ent in doc.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}")
    if ent._.kb_id_:
        print(f"Linked ID: {ent._.kb_id_}")
        print(f"Linked KB Entry: {ent._.kb_}")
    else:
        print("No link found")
    print()

# Disable the EntityLinker component if needed
nlp.remove_pipe("entity_linker")
