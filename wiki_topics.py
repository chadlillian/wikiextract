#!/home/chad/anaconda/bin/python

from sklearn.decomposition import TruncatedSVD
from multiprocessing import Pool
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import vstack
import pandas as pd
import numpy as np
from scipy.sparse import coo_array
import scipy.sparse as scsp
import glob
import time
import pickle

# loops through dfs with lemmas
#   reads in a single df with lemmas at a time
#   parallelizes countvectorizer chunks from each df
#   recombines countvectorizers back into single df
# combines all countvectorizers into large df
# runs tfidf
# runs pca

class ptfidf:
    def __init__(self):
        None
        self.vocab = [] 
        self.counts = []
        self.titles = []

    #   count lemmas from each chunk
    def readDFSection(self,df,lemmacol,titlecol):
        vectorizer = CountVectorizer()
        corpus = df[lemmacol]
    
        count = vectorizer.fit_transform(corpus)
        vocab = vectorizer.vocabulary_

        return vocab,count,df[titlecol]
    
    def parallelMergeCountVectorizer(self,allvocabd,allvocabl,thisvocab,thiscount):
        lenvc = len(thisvocab)
        lenv = len(allvocabl)
        row = []
        col = []
        val = []
        for word in list(thisvocab.keys()):
            if word in allvocabl:
                row.append(thisvocab[word])
                col.append(allvocabd[word])
                val.append(1)
        tr = coo_array((val,(row,col)),shape=(lenvc,lenv))
        thiscount_ = thiscount@tr

        return thiscount_

    def prepVocabForParallel(self,vocabs):
        vocab_ = list(set().union(*vocabs))
        vocab_l = []
        for word in vocab_:
            if word.isascii() and word.isalpha():
                vocab_l.append(word)
        vocab_d = {w:i for i,w in enumerate(vocab_l)}
        return vocab_l,vocab_d

    def postVocabForParallel(self,sats,vocab_d,save=False):
        counts = scsp.vstack(sats)
        self.cv = CountVectorizer(vocabulary=vocab_d)
        if save:
            with open('vectorizer.pkl','wb') as f:
                pickle.dump(self.cv,f)
        return vocab_d,counts

    #   this adds (combines) two count vectorizers
    #   it really combines the 
    def addCountVectorizers(self,vocabs,counts,save=False):
        vocab_l,vocab_d = self.prepVocabForParallel(vocabs)

        chunks = [(vocab_d,vocab_l,vc,sa) for vc,sa in zip(vocabs,counts)]
        with Pool() as pool:
            z = pool.starmap(self.parallelMergeCountVectorizer, chunks)
        sats = z

        vocab_d,counts = self.postVocabForParallel(z,vocab_d,save=False)
        return vocab_d,counts

    #   count lemmas from each wiki page, create and save a countvectorizer and a word frequency csv file
    def getLemmaCounts(self,df,lemmacol,titlecol,numprocs):
        N = numprocs
        df['group'] = np.arange(df.shape[0])//int(np.ceil(df.shape[0]/N))
        dfg = df.groupby('group')
        dfgp = [(x[1],lemmacol,titlecol) for x in dfg]

        with Pool() as pool:
            z = pool.starmap(self.readDFSection, dfgp)
        
        vocabs = [zz[0] for zz in z]
        counts = [zz[1] for zz in z]
        titles = pd.concat([zz[2] for zz in z])
        
        vocab,counts = self.addCountVectorizers(vocabs,counts)

        self.titles.append(titles)
        self.vocab.append(vocab)
        self.counts.append(counts)

    def finishLemmaCounts(self):
        self.vocab,self.counts = self.addCountVectorizers(self.vocab,self.counts,save=True)
        self.titles = pd.concat(self.titles).reset_index(drop=True)

        tft = TfidfTransformer()
        allcounts = tft.fit_transform(self.counts)

        clf = TruncatedSVD(10)
        ah = clf.fit_transform(allcounts)

        pcacolnames=['pca_{i}'.format(i=i) for i in range(ah.shape[1])]
        ret = pd.DataFrame(ah,columns=pcacolnames)
        ret['titles'] = self.titles
        ret = ret[['titles']+pcacolnames]

        return ret #self.titles,ah

if __name__ == '__main__':
    n_chunks = 8  # Adjust as needed

    testfilenames = glob.glob('test_*.csv')
    z = ptfidf()

    for fn in testfilenames:
        print('*'*88)
        print(fn)
        df = pd.read_csv(fn)
        z.getLemmaCounts(df,'lemmas','title',numprocs=3)

    pcadf = z.finishLemmaCounts()
    print(pcadf)
    #print(titles)
    #print(arr)
