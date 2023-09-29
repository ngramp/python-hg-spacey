import spacy
import cupy as cp

from records import iterator

# Load a pre-trained spaCy model with NER
# nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_sm", disable=["ner"])
nlp.enable_pipe("ner")

text = "Facebook and LinkedIn are social media companies. David Fry works at Facebook, and Kole Calhoun works at LinkedIn."

# Process the text with spaCy
for doc in nlp.pipe(iterator):
    # Extract recognized entities
    for ent in doc.ents:
        print(f"Entity: {ent.text}, Label: {ent.label_}")


# Custom entity linking (example)
entity_mapping = {
    "Fry": "David Fry",  # Link "Fry" to "David Fry"
    "LinkedIn": "LinkedIn Corporation",  # Example: Link "LinkedIn" to a full entity name
}

# Resolve entities with the same name based on custom mapping
resolved_entities = [entity_mapping.get(ent.text, ent.text) for ent in doc.ents]

print("Resolved Entities:")
for resolved_entity in resolved_entities:
    print(resolved_entity)
