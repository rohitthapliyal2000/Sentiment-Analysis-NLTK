import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle

def create_word_features(words):

	no_more_discrete = []
	for i in words:
		no_more_discrete.append(i[0].lower())

	useful_words = [word for word in no_more_discrete if word not in stopwords.words("english")]
	my_dict = dict([(word, True) for word in useful_words])
	return my_dict

lemmatizer = WordNetLemmatizer()

neg_reviews = []
pos_reviews = []

for fileid in movie_reviews.fileids('neg'):
	words_ = movie_reviews.words(fileid)
	words = []


	for i in words_:
		words.append(word_tokenize(i))

	for i in words_:
		alt_word = lemmatizer.lemmatize(i, pos = "a")
		if(alt_word != i):
			words.append(alt_word)

	neg_reviews.append((create_word_features(words), "negative"))


for fileid in movie_reviews.fileids('pos'):
	words_ = movie_reviews.words(fileid)
	words = []

	for i in words_:
		words.append(word_tokenize(i))

	for i in  words_:
		alt_word = lemmatizer.lemmatize(i, pos = "a")
		if(alt_word != i):
			words.append(alt_word)

	pos_reviews.append((create_word_features(words), "positive"))

train_set = neg_reviews[:750] + pos_reviews[:750]
test_set = neg_reviews[750:] + pos_reviews[750:]

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
