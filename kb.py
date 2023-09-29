from spacy.kb import KnowledgeBase
from spacy.vocab import Vocab
from spacy.lang.en import English
from spacy.kb import InMemoryLookupKB
import spacy

# Download and install the 'en_core_web_lg' model
spacy.cli.download("en_core_web_lg")
nlp = spacy.load("en_core_web_lg")
doc = nlp("Douglas Adams wrote 'The Hitchhiker's Guide to the Galaxy'.")
vocab = nlp.vocab
kb = InMemoryLookupKB(vocab=vocab, entity_vector_length=64)

candidates = kb.get_candidates(doc[0:5])
print(candidates)
