from transformers import AutoTokenizer, AutoModelForTokenClassification, logging
from transformers import pipeline

import records
# from linking import link_to_wikipedia

logging.set_verbosity_error()

print("Setting up tokeniser...")
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
print("Setting up Model...")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
model.to_bettertransformer()
# Initialize the NER pipeline with a pre-trained model
print("Setting up NER pipeline...")
ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    device="cuda:0",
    use_fast=True,
    aggregation_strategy="max",
)


def bert_detect():
    for out in ner_pipeline(records.iterator, batch_size=1):
        entities = []
        for item in out:
            word = item["word"]
            entities.append(word)
        # link_to_wikipedia(entities)


# Example usage
if __name__ == "__main__":
    bert_detect()
    # flair_detect()
