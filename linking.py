from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize


# Load your pre-trained Word2Vec model
word2vec_model = Word2Vec.load(
    "/home/gram/PycharmProjects/pythonProject/en_1000_no_stem/en.model"
)

# Access the word vectors
word_vectors = word2vec_model.wv


def link_to_wikipedia(entities):
    preprocessed_entities = [word_tokenize(entity.lower()) for entity in entities]

    corresponding_articles = []

    for entity, tokens in zip(entities, preprocessed_entities):
        # Initialize a list to store similarities
        similarities = []

        for token in tokens:
            try:
                # Calculate cosine similarity between the token and words in the model
                similar_words = word_vectors.most_similar(token, topn=1)
                # Add the similarity score to the list
                similarities.append(similar_words[0][1])
            except KeyError:
                # Handle cases where a token is not in the vocabulary
                similarities.append(0.0)

        # Calculate the average similarity for the entity
        avg_similarity = sum(similarities) / len(similarities)
        corresponding_articles.append((entity, avg_similarity))

    # Sort entities by average similarity (descending order)
    corresponding_articles.sort(key=lambda x: x[1], reverse=True)

    # Print the results
    for entity, avg_similarity in corresponding_articles:
        print(f"Entity: {entity}, Average Similarity: {avg_similarity:.4f}")
