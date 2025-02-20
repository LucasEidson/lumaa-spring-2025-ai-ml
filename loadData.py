import csv
import math
import pickle
import re

DATA_PATH = 'movie_plots.csv' #from https://www.kaggle.com/datasets/kartikeychauhan/movie-plots
PREPROCESSED_PATH = 'preprocessed.csv'
NUM_MOVIES = 500
RELEVANT_COLUMNS = "title", "year", "genre", "plot", "imdb_rating",

def main():
    loadData()
    movies = createMovieList()
    termScores = getTFIDFs(movies, getIDFs(movies))
    with open("tfidf_data.pkl", "wb") as file:
        pickle.dump(termScores, file)

def loadData():
    with open(DATA_PATH, newline='', encoding='latin-1') as dataset, open(PREPROCESSED_PATH, mode='w') as preprocessed:
        reader = csv.reader(dataset, delimiter=',')
        writer = csv.writer(preprocessed, delimiter=',')
        columns = next(reader)
        relevantColumnIndicies = [columns.index(i) for i in RELEVANT_COLUMNS]
        writer.writerow(columns[i] for i in relevantColumnIndicies)
        for i in range(NUM_MOVIES):
            line = next(reader)
            #Only get movies with plot summaries:
            if(line[relevantColumnIndicies[3]] != ' ' and line[relevantColumnIndicies[3]] != ''):
                writer.writerow(line[i] for i in relevantColumnIndicies)

def createMovieList():
    #Creates a list of movies in the preprocessed dataset and calculates their TF-IDF Vectors
    with open(PREPROCESSED_PATH, newline='', encoding='latin-1') as preprocessed:
        reader = csv.reader(preprocessed)
        next(reader) #skips first line with column titles
        movies = []
        for line in reader:
            newMovie = movie(*line)
            movies.append(newMovie)
    return movies

def getIDFs(movies):
    numDocs = len(movies)
    corpusTerms = []
    termIDFs = dict()
    for m in movies:
        visitedTerms = []
        for t in m.TF:
            if t in visitedTerms:
                continue
            else:
                visitedTerms.append(t)
            if not t in termIDFs:
                termIDFs[t] = 1
            else:
                termIDFs[t] += 1
            if t not in corpusTerms:
                corpusTerms.append(t)
    for term in corpusTerms:
        termIDFs[term] = math.log(numDocs/termIDFs[term])
    return(termIDFs)

def getTFIDFs(movies, termIDFs):
    termScores = [] #will be a list of dictionaries representing TFIDF Values for terms
    index = 0 #index of movie
    for m in movies:
        movieTFIDFs = dict()
        for t in termIDFs: #ensures that every dict has a TF-IDF value for every term
            if t in m.TF:
                TF = m.TF[t]
                IDF = termIDFs[t]
                movieTFIDFs[t] = TF * IDF
            else:
                movieTFIDFs[t] = 0
        termScores.append(movieTFIDFs)
        index += 1
    return(termScores)

class movie:
    def __init__(self, title, releaseYear, genre, plot, rating):
        self.title = title
        self.releaseYear = releaseYear
        self.genre = genre
        self.plot = plot
        self.rating = rating
        self.TF = self.getTF()

    def getTF(self):
        terms = self.title.lower().split() + self.plot.lower().split()
        genreTerms = self.genre.lower().split()
        numTerms = len(terms)
        TF = dict()
        for term in terms:
            term = re.sub(r'[^\w\s]','', term) #remove punctuation
            if not term in TF:
                TF[term] = 1
            else:
                TF[term] += 1
        #Genre Terms are weighted to be more important (counted twice) than plot summary terms
        for term in genreTerms:
            term = re.sub(r'[^\w\s]','', term) #remove punctuation
            if not term in TF:
                TF[term] = 2
            else:
                TF[term] += 2
        for k in TF:
            TF[k] /= numTerms
        return TF


if __name__ == "__main__":
    main()