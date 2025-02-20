import pickle
import numpy as np
import re
from loadData import createMovieList

PREPROCESSED_PATH = 'preprocessed.csv'
TOP_N = 3 #number of recommendations

def main():
    #query is a single sentence so IDF Vector is just 1
    query = input("What kind of movies would you like me to find?\n")
    #the result of the query will be more accurate if you avoid things like "i like movies with ...",
    #and instead just say "action comedy set in space"
    query = query.lower() 
    query = re.sub(r'[^\w\s]','', query) #remove punctuation
    with open("tfidf_data.pkl", "rb") as file:
        termScores = pickle.load(file)
    movies = createMovieList()
    queryTF = getTF(query)
    vocab = getVocab(queryTF, termScores)
    print("Here are my top ", TOP_N, " recommendations for you:")
    similarities, movies = sortBySimilarity(termScores, queryTF, vocab, movies)
    for i in range(TOP_N): #movies also stores info like genre, rating, and plot description
        print("Movie Title: ", movies[i].title)
        print("TF-IDF Similarity Score: ", similarities[i])
        print()


def sortBySimilarity(termScores, queryTF, vocab, movies):
    similarities = []
    for movieDict in termScores:
        similarities.append(cosineSimilarity(queryTF, movieDict, vocab))
    similarities = zip(similarities, range(0, len(movies)))
    similarities = sorted(similarities, reverse=True)
    similarities, moviesIndicies = list(zip(*similarities))
    #Rearrange movies list to be sorted by similarity
    temp = movies.copy()
    j = 0
    for i in moviesIndicies:
        movies[j] = temp[i]
        j += 1 
    return(similarities, movies)

def getTF(query):
    terms = query.split()
    numTerms = len(terms)
    TF = dict()
    for term in terms:
        if not term in TF:
            TF[term] = 1
        else:
            TF[term] += 1
    for k in TF:
        TF[k] /= numTerms
    return TF

def getVocab(query, termScores):
    vocab = []
    for t in termScores[0]:
        vocab.append(t)
    for t in query:
        if t not in vocab:
            vocab.append(t)
    return vocab

def cosineSimilarity(queryTF, termTF, vocab):
    movieVector = np.array([termTF.get(term, 0) for term in vocab])
    queryVector = np.array([queryTF.get(term, 0) for term in vocab])
    dotProduct = np.dot(movieVector, queryVector)
    movieNorm = np.linalg.norm(movieVector)
    queryNorm = np.linalg.norm(queryVector)
    if movieNorm == 0 or queryNorm == 0:
        return 0
    else:
        return (dotProduct/(movieNorm * queryNorm))

if __name__ == "__main__":
    main()
