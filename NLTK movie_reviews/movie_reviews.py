import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import itertools
import pickle

def create_word_features(words):
	return dict([(word, True) for word in words])

neg_reviews = []
pos_reviews = []

negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

print("1")

neg_reviews = [(create_word_features(movie_reviews.words(fileids = [inn])), 'negative') for inn in negids]
pos_reviews = [(create_word_features(movie_reviews.words(fileids = [inn])), 'positive') for inn in posids]

print("2")

neg_cutoff = 750
pos_cutoff = 750

train_set = neg_reviews[:neg_cutoff] + pos_reviews[:pos_cutoff]
test_set = neg_reviews[neg_cutoff:] + pos_reviews[pos_cutoff:]

#Run next 4 instructions if you're running the script for first time 
classifier = NaiveBayesClassifier.train(train_set)
classify_buffer = open('movie_reviews.pickle', 'wb')
pickle.dump(classifier, classify_buffer)
classify_buffer.close()
#Comment above 4 instructions if you've run the script once

#Run next 3 instructions if you're running the script second time onwards
#classify_buffer = open('movie_reviews.pickle', 'rb')
#classifier = pickle.load(classify_buffer)
#classify_buffer.close()
#Comment aboce 3 instructions while running the script for first time

print("Accuracy is : ", nltk.classify.util.accuracy(classifier, test_set) * 100)
