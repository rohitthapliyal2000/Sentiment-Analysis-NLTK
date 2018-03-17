# Sentiment-Analysis-NLTK

## Basic ALgorithm :

NaiveBayesClassifier was used as the opinion classifier.

The dataset included a list of sentences.

The sentences was converted to form bag of words.

75% of the corpora was used as the Training set.

Rest 25% was used to test the Classifier module.

## Modifying the algorithm :

Markup : 

* Using first __'N'__ frequent words can increase the accuracy.

__Note :__ N should be changed along a certain range to check the peak point.

* There can be sentences like : 

*This is not good*

Bigrams (pair of words) can be used to include __{'not', 'good'}__ in the bag of words.

Using bigrams can exponentially increase the algorithmic accuracy.

* For opinion mining, including emoticons can make the Classifier a lot better.


## Analysis of movie_reviews corpora :

Accuracy on modified algorithm : __72.8

Accuracy with algorithm including bigrams : __84.8

## Analysis of twitter_samples corpora :

Accuracy with algorithm excluding emoticons : __76.9

Accuracy on modified algorithm : __97.35

Accuracy with algorithm including bigrams : __99.1
