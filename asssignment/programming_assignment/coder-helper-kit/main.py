from classifier import load_data,tokenize, feature_extractor, classifier_agent
from collections import Counter

import numpy as np

import spacy
nlp = spacy.load('en_core_web_sm')

from nltk.corpus import stopwords, words
from nltk.stem import PorterStemmer

porter_stemmer = PorterStemmer()

s=set(stopwords.words('english'))

txt="a long string of text about him and her"
print(list(filter(lambda w: not w in s,txt.split())))


def main():
    print("Creating a classifier agent:")

    with open('data/vocab.txt') as file:
        reading = file.readlines()
        vocab_list = [item.strip() for item in reading]
        vocab_dict = {item: i for i, item in enumerate(vocab_list)}

    print("Loading and processing data ...")

    sentences_pos = load_data("data/training_pos.txt")
    sentences_neg = load_data("data/training_neg.txt")

    train_sentences = sentences_pos[:3000] + sentences_neg[:3000]

    train_labels = [1 for i in range(len(sentences_pos[:3000]))] + [0 for i in range(len(sentences_neg[:3000]))]

    sentences_pos = load_data("data/test_pos_public.txt")
    sentences_neg = load_data("data/test_neg_public.txt")
    test_sentences = sentences_pos + sentences_neg
    test_labels = [1 for i in range(len(sentences_pos))] + [0 for i in range(len(sentences_neg))]
    formatted_vocab_list = []
    
    formatted_vocab_list = list(filter(lambda w: not w in s,vocab_list))
    formatted_vocab_list = [porter_stemmer.stem(word) for word in formatted_vocab_list]
    filter_s = set(words.words())
    # print('after stemming', len(list(set(formatted_vocab_list))))
    
    
    # formatted_vocab_list = list(filter(lambda w: not w in filter_s,formatted_vocab_list))
    formatted_vocab_list = list(filter(lambda w: not w in filter_s,vocab_list))
    formatted_vocab_dict = {item: i for i, item in enumerate(formatted_vocab_list)}
    
    
    
    # feat_map = feature_extractor(vocab_list, tokenize)
    feat_map = feature_extractor(formatted_vocab_list, tokenize)
    
    print('after feature extractor')
    
    print('feature map', feat_map)
    # You many replace this with a different feature extractor
    # word_freq = 

    # feat_map = tfidf_extractor(vocab_list, tokenize, word_freq)

    # train with GD
    niter = 100
    print("Training using GD for ", niter, "iterations.")
    # d = len(vocab_list)
    d = len(formatted_vocab_list)
    params = np.array([0.0 for i in range(d)])
    classifier1 = classifier_agent(feat_map,params)
    classifier1.train_gd(train_sentences,train_labels,niter,0.01)




    # train with SGD
    nepoch = 10
    print("Training using SGD for ", nepoch, "data passes.")
    # d = len(vocab_list)
    d = len(formatted_vocab_list)
    params = np.array([0.0 for i in range(d)])
    classifier2 = classifier_agent(feat_map, params)
    classifier2.train_sgd(train_sentences, train_labels, nepoch, 0.001)


    err1 = classifier1.eval_model(test_sentences,test_labels)
    err2 = classifier2.eval_model(test_sentences,test_labels)

    print('GD: test err = ', err1,
          'SGD: test err = ', err2)
    print('SGD: test err = ', err2)


if __name__ == "__main__":
    main()