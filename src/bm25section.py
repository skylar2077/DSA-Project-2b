import string
import pickle as pic
import math as m
import numpy as np


class bm25O :
    def __init__(self, database, k1 = 1.2, b = 0.75, e = 0.15):
        self.databaseSize = 0
        self.database = database # 2D array of all items & their words
        self.k1 = k1 # used to approximate a poisson distribution in a parametric function
        self.b = b # used in document length normalization
        self.e = e # used for minimums in IDF
        self.avgdl = 0 # average length of documents in the database
        self.IDF = {} # Inverse Document Frequency: see what terms are rare in the database, and value them more
        self.dl = [] # list of the lengths of docs in database
        self.frequencies = [] # list of dictionaries containing word frequency pairs for each document
        self.docNum = {} # dictionary of pairs for words & number of documents that have that word

        # initialization


        # this section sets up the avgdl, dl, frequencies, databaseSize, and docNum attributes
        totalLen = 0
        for item in database:
            self.dl.append(len(item))
            totalLen += len(item)


            docFreqs = {} # dictionary of word frequency pairs within the document
            for word in item:

                if not word in docFreqs:
                    docFreqs[word] = 1
                else:
                    docFreqs[word] += 1

            self.frequencies.append(docFreqs)


            for word, frequency in docFreqs.items():
                # ups number of docs that contain the word
                if not word in self.docNum:
                    self.docNum[word] = 1
                else:
                    self.docNum[word]+=1

            self.databaseSize += 1 # ups count once for every item in the database



        self.avgdl = totalLen / self.databaseSize # average length of documents = total word number / number of items



        # this section sets up the IDF attribute, running each word in the database through the function & storing it
        # in the object. This uses the Okapi version of the algorithm
        totalIDF = 0 # summing all IDF values to get the average, then taking the product with epsilon for the floor
        IDFlt0 = [] # list of all values < 0 to be floored by the epsilon value

        for word, freq in self.docNum.items():
            tempIDF = m.log(self.databaseSize - freq + 0.5) - m.log(freq + 0.5)
            # b/c of log properties, equivalent to log((size-freq+.5)/(freq+0.5)) and it looks nicer

            if tempIDF < 0:
                IDFlt0.append(word)
            totalIDF+= tempIDF
            self.IDF[word] = tempIDF

        avgIDF = totalIDF / len(self.IDF)
        IDFfloor = self.e * avgIDF
        for word in IDFlt0:
            self.IDF[word] = IDFfloor



    def getScores(self, query):
        # These lists will be having alot of math done on them, so I decided to use numpy for this function, because
        # its a little faster than python lists, but also better for when alot of math needs to be done.
        scores = np.zeros(self.databaseSize) # creates a numpy array of all zeros, one for each item in the database
        tempdl = np.array(self.dl) # makes a numpy array based on the dl. This is because I don't know how lists plus
        # numpy arrays work when interacting with eachother
      

        for term in query:
            # here the (dic.get(term) or 0) is used to ensure that an integer is used in the function
            tf = np.array([(item.get(term) or 0) for item in self.frequencies])
            scores += ((self.IDF.get(term) or 0) *
                    (tf * (self.k1 + 1)) /
                    (tf + self.k1 * (1 - self.b + self.b * tempdl / self.avgdl)))

        return scores.tolist()



