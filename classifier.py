import time

from transformers import AutoTokenizer, AutoModelForTokenClassification, logging
from transformers import pipeline

import records

logging.set_verbosity_error()

print("Setting up Classifier...")
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device="cuda:0",
    use_fast=True,
)
# classifier = pipeline(
#     "zero-shot-classification",
#     model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
#     device="cuda:0",
#     use_fast=True,
# )


def clas():
    candidate_labels = [
        "global health",
        "management",
        "equity equality",
        "ESG",
        "governance",
        "climate change",
    ]
    i = 0
    start = time.time()
    for out in classifier(records.iterator, candidate_labels):
        print(out["scores"])
        i = i + 1
        if i % 100 == 0:
            now = time.time()
            elapsed = now - start
            average = elapsed / 100
            print("Average time per record: {:.4f} seconds".format(average))
            start = now


# Example usage
if __name__ == "__main__":
    clas()
    # flair_detect()
