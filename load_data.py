"""
This Module reads and processes movie data from a CSV File. It creates two files:
    A CSV file containing only relevant movie data from the original CSV File.
    A .pkl file containing a list of dicts representing TFIDF Vectors for each movie in the processed CSV.

Classes:
    Movie: A class representing a movie.

Functions:
    load_data(): Creates new processed CSV file at PREPROCESSED_PATH from original CSV file at DATA_PATH.
    create_movie_list(): Returns a list of Movie objects based on information from the processed movie data.
    get_idfs(): Returns a dict of IDF values for terms in the movies plot, title, and genres.
    get_tfidfs(): Returns a list of TFIDF vectors, represented by dicts, for each movie.
"""

import csv
import math
import pickle
import re

DATA_PATH = "movie_plots500.csv"  # altered from https://www.kaggle.com/datasets/kartikeychauhan/movie-plots
PREPROCESSED_PATH = "preprocessed.csv"
NUM_MOVIES = 500
RELEVANT_COLUMNS = (
    "title",
    "year",
    "genre",
    "plot",
    "imdb_rating",
)


def main():
    """Main function ran when executing load_data.py from console."""
    load_data()
    movies = create_movie_list()
    term_scores = get_tfidfs(movies, get_idfs(movies))
    with open("tfidf_data.pkl", "wb") as file:
        pickle.dump(term_scores, file)


def load_data():
    """Create a new CSV File containing only relevant columns and a limited number of rows."""
    with open(DATA_PATH, newline="", encoding="latin-1") as dataset, open(
        PREPROCESSED_PATH, mode="w"
    ) as preprocessed:
        reader = csv.reader(dataset, delimiter=",")
        writer = csv.writer(preprocessed, delimiter=",")
        columns = next(reader)
        relevant_column_indicies = [columns.index(i) for i in RELEVANT_COLUMNS]
        writer.writerow(columns[i] for i in relevant_column_indicies)
        for i in range(NUM_MOVIES):
            line = next(reader)
            # Only get movies with plot summaries:
            if (
                line[relevant_column_indicies[3]] != " "
                and line[relevant_column_indicies[3]] != ""
            ):
                writer.writerow(line[i] for i in relevant_column_indicies)


def create_movie_list():
    """Create and return a list of Movie objects from processed csv data."""
    with open(PREPROCESSED_PATH, newline="", encoding="latin-1") as preprocessed:
        reader = csv.reader(preprocessed)
        next(reader)  # Skips first line with column titles.
        movies = []
        for line in reader:
            new_movie = Movie(*line)
            movies.append(new_movie)
    return movies


def get_idfs(movies):
    """Return a single dictionary of IDF values for every term in every Movie object in the movies list."""
    num_docs = len(movies)
    corpus_terms = []
    term_idfs = dict()
    for m in movies:
        visited_terms = []
        for t in m.tf:
            if t in visited_terms:
                continue
            else:
                visited_terms.append(t)
            if not t in term_idfs:
                term_idfs[t] = 1
            else:
                term_idfs[t] += 1
            if t not in corpus_terms:
                corpus_terms.append(t)
    for term in corpus_terms:
        term_idfs[term] = math.log(num_docs / term_idfs[term])
    return term_idfs


def get_tfidfs(movies, term_idfs):
    """
    Return a list with a dictionary for each movie, containing TF-IDF values for every term.

    Args:
        movies: a list of Movie objects
        term_idfs: a dict of IDF values for terms in each movie in Movies

    Returns:
        termScores: a list of dicts, where each dict represents TFIDF values for every term in a movie
    """
    term_scores = []
    index = 0
    for m in movies:
        movie_tfidfs = dict()
        for (
            t
        ) in term_idfs:  # Ensures that every dict has a TF-IDF value for every term.
            if t in m.tf:
                tf = m.tf[t]
                idf = term_idfs[t]
                movie_tfidfs[t] = tf * idf
            else:
                movie_tfidfs[t] = 0
        term_scores.append(movie_tfidfs)
        index += 1
    return term_scores


class Movie:
    """
    A class to represent a movie.

    Attributes:
        title (String): The title of the movie.
        release_year (String): The year the movie was released.
        genre (List of Strings): List of genres describing the movie.
        plot (String): A plot summary of the movie.
        rating (String): The IMDB Rating of the movie.

    Methods:
        get_tf(): Calculates and returns as a dict the term frequency of each term in the genre, title, and plot of the movie.
    """

    def __init__(self, title, release_year, genre, plot, rating):
        self.title = title
        self.releaseYear = release_year
        self.genre = genre
        self.plot = plot
        self.rating = rating
        self.tf = self.get_tf()

    def get_tf(self):
        """Calculate and return as a dict the term frequency of each term in the genre, title, and plot of the movie. Genre terms are weighted 2x."""
        terms = self.title.lower().split() + self.plot.lower().split()
        genre_terms = self.genre.lower().split()
        num_terms = len(terms)
        tf = dict()
        for term in terms:
            term = re.sub(r"[^\w\s]", "", term)  # Removes punctuation.
            if not term in tf:
                tf[term] = 1
            else:
                tf[term] += 1
        # Genre Terms are weighted to be more important (counted twice) than plot summary terms.
        for term in genre_terms:
            term = re.sub(r"[^\w\s]", "", term)  # Removes punctation.
            if not term in tf:
                tf[term] = 2
            else:
                tf[term] += 2
        for k in tf:
            tf[k] /= num_terms
        return tf


if __name__ == "__main__":
    main()
