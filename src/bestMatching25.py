import string
import pickle
import pandas as pd
import io
from bm25section import bm25O

# def simplify(instring, string):
#     return instring.translate(str.maketrans('', '', string.punctuation)).lower()


'''
    This block of commented out code has to do with the way I created the pickle file and BM25 objects. It has no purpose anymore.
'''
# df = pd.read_csv("GoodReads_100k_books.csv", usecols=["author", "desc", "genre", "img", "isbn", "link", "pages", "rating", "reviews", "title", "totalratings"])

# nthrow = df.iloc[30]
# print(nthrow)

# titles = df["title"]
# simple_titles = []
# for title in titles:
#     if (not (type(title) == type("string"))):
#         simple_titles.append("N/A")
#         continue
#     simple_titles.append(simplify(title, string))
#
#
# authors = df["author"]
# simple_authors = []
# for item in authors:
#     if (not (type(item) == type("string"))):
#         simple_authors.append("N/A")
#         continue
#     simple_authors.append(simplify(item, string))
#
#
# descs = df["desc"]
# simple_descs = []
# for item in descs:
#     if (not (type(item) == type("string"))):
#         simple_descs.append("N/A")
#         continue
#     simple_descs.append(simplify(item, string))
#
#
# genres = df["genre"]
# simple_genres = []
# for item in genres:
#     if (not (type(item) == type("string"))):
#         simple_genres.append("N/A")
#         continue
#     simple_genres.append(item.replace(",", " ").lower())
#
#
# ratings = df["rating"]
# isbns = df["isbn"]
#
# tokenized_titles = [item.split(" ") for item in simple_titles]
# tokenized_authors = [item.split(" ") for item in simple_authors]
# tokenized_descs = [item.split(" ") for item in simple_descs]
# tokenized_genres = [item.split(" ") for item in simple_genres]
#
# bmtitles = bm25O(tokenized_titles)
# bmauthors = bm25O(tokenized_authors)
# bmdescs = bm25O(tokenized_descs)
# bmgenres = bm25O(tokenized_genres)

# bmObj = [bmtitles, bmauthors, bmdescs, bmgenres]

# pickle.dump(bmObj, open("data.pkl", "wb"))


def loadPickle():
    temp = pickle.load(open("../data/data.pkl", "rb"))
    return temp

def topNBook(inDict, query, n, inpickle):

    titleVec = inpickle[0].getScores(query)
    authorsVec = inpickle[1].getScores(query)
    descsVec = inpickle[2].getScores(query)
    genresVec = inpickle[3].getScores(query)


    topN = []
    for i in range(100000):
        temp = (titleVec[i] * inDict["title"] +
                   authorsVec[i] * inDict["authors"] +
                   descsVec[i] * inDict["descs"] +
                   genresVec[i] * inDict["genres"])
        topN.append(temp)



    top = np.argsort(topN)[::-1][:n]
    out = df.loc[top, ["author", "desc", "genre", "img", "isbn", "link", "pages", "rating", "reviews", "title", "totalratings"]].copy()
    return out.reset_index(drop=True)



