# Sentiment-Analysis-NLTK

## Basic Algorithm:

NaiveBayesClassifier was used as the opinion classifier.

The dataset included a list of sentences.

The sentences was converted to form bag of words.

75% of the corpora was used as the Training set.

Rest 25% was used to test the Classifier module.

*NaiveBayesClassifier :* https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c

## Modifying the algorithm:

*   Using first __'N'__ frequent words can increase the accuracy.

    __Note :__ N should be changed along a certain range to check the peak point.

*   There can be sentences like : 

    *This is not good*

    Bigrams (pair of words) can be used to include __{'not', 'good'}__ in the bag of words.

    Using bigrams can exponentially increase the algorithmic accuracy.

*   For opinion mining, including emoticons can make the Classifier a lot better.

*   Pickle can be used to save the training model once and for all.
    
    The algorithm can be trained once and classifying the data will take no time henceforth.


## Analysis of movie_reviews corpora:

Accuracy on modified algorithm: __72.8__

Accuracy with algorithm including bigrams: __84.8__


## Analysis of twitter_samples corpora:

Accuracy with algorithm excluding emoticons: __76.9__

Accuracy on modified algorithm: __97.35__

Accuracy with algorithm including bigrams: __99.1__ 
