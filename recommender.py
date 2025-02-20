"""
This Module uses a preprocessed CSV file of movies and a .pkl file containg TFIDF Vectors to recommend movies
based on cosine similarity of TFIDF Vectors.

Functions:
    get_tf(): Returns a dictionary of term frequencies in a string.
    get_vocab(): Returns a list of all terms(keys) in input dicts.
    cosine_similarity(): Returns the cosine similarity between two dicts where each key is a term and each value is its TFIDF value.
    sort_by_similarity(): Uses cosine similarity to calculate and sort each movie based on its similarity to a query.
"""

import pickle
import numpy as np
import re
from load_data import create_movie_list

PREPROCESSED_PATH = "preprocessed.csv"
TOP_N = 3  # Number of recommendations


def main():
    """Main function ran when executing load_data.py from console. Get query as input from console."""
    # Query is a single sentence so IDF value is just 1.
    query = input("What kind of movies would you like me to find?\n")
    query = query.lower()
    query = re.sub(r"[^\w\s]", "", query)  # Removes punctuation.
    with open("tfidf_data.pkl", "rb") as file:
        term_scores = pickle.load(file)
    movies = create_movie_list()
    query_tf = get_tf(query)
    vocab = get_vocab(query_tf, term_scores)
    print("Here are my top ", TOP_N, " recommendations for you:")
    similarities, movies = sort_by_similarity(term_scores, query_tf, vocab, movies)
    for i in range(
        TOP_N
    ):  # Movies also stores info like genre, rating, and plot description.
        print("Movie Title: ", movies[i].title)
        print("TF-IDF Similarity Score: ", similarities[i])
        print()


def sort_by_similarity(term_scores, query_tf, vocab, movies):
    """
    Use cosine similarity to calculate and sort each movie based on its similarity to a query.

    Args:
        term_scores: A list of dicts with each dict representing the TFIDF vector of a movie.
        query_tf: A dict representing term frequency of terms in query where key is term and value is frequency.
        vocab: A list of all terms in each dict of term_scores and query_tf.
        movies: A list of Movie objects where movies[i] corresponds to term_scores[i].

    Returns:
        similarities: list of sorted(high to low) cosine similarity values for each movie compared to the query.
        movies: list of Movie objects sorted by similarity values in similarities (similarities[i] corresponds to
            the cosine similarity of movies[i] and query_tf).
    """
    similarities = []
    for movie_dict in term_scores:
        similarities.append(cosine_similarity(query_tf, movie_dict, vocab))
    similarities = zip(similarities, range(0, len(movies)))
    similarities = sorted(similarities, reverse=True)
    similarities, movies_indicies = list(zip(*similarities))
    # Rearrange movies list to be sorted by similarity:
    temp = movies.copy()
    j = 0
    for i in movies_indicies:
        movies[j] = temp[i]
        j += 1
    return (similarities, movies)


def get_tf(query):
    """Return a dict of term frequencies in the query(String) where key is term and value is frequency."""
    terms = query.split()
    num_terms = len(terms)
    tf = dict()
    for term in terms:
        if not term in tf:
            tf[term] = 1
        else:
            tf[term] += 1
    for k in tf:
        tf[k] /= num_terms
    return tf


def get_vocab(query, term_scores):
    """Return a list of all terms in query and each dict of term_scores where term_scores is a list of dicts."""
    vocab = []
    for t in term_scores[0]:  # Each dict of term_scores already has the same vocab.
        vocab.append(t)
    for t in query:
        if t not in vocab:
            vocab.append(t)
    return vocab


def cosine_similarity(query_tf, term_tf, vocab):
    """
    Return the cosine similarity tbetween two dicts based on a list of all terms in both dicts.

    Args:
        query_tf, term_tf: A dict where key is term and value is frequency.
        vocab: a list of all terms in query_tf and term_tf.
    """
    movie_vector = np.array([term_tf.get(term, 0) for term in vocab])
    query_vector = np.array([query_tf.get(term, 0) for term in vocab])
    dot_product = np.dot(movie_vector, query_vector)
    movie_norm = np.linalg.norm(movie_vector)
    query_norm = np.linalg.norm(query_vector)
    if movie_norm == 0 or query_norm == 0:
        return 0
    else:
        return dot_product / (movie_norm * query_norm)


if __name__ == "__main__":
    main()
