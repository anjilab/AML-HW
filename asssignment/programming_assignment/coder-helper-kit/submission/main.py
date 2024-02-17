from classifier import load_data,tokenize, feature_extractor, classifier_agent, compute_word_idf, tfidf_extractor
from collections import Counter

import numpy as np

import spacy
nlp = spacy.load('en_core_web_sm')

from nltk.corpus import stopwords, words
from nltk.stem import PorterStemmer

porter_stemmer = PorterStemmer()

s=set(stopwords.words('english'))



def main():
    print("Creating a classifier agent:")

    with open('data/vocab.txt') as file:
        reading = file.readlines()
        vocab_list = [item.strip() for item in reading]
        vocab_dict = {item: i for i, item in enumerate(vocab_list)}

    print("Loading and processing data ...")

    sentences_pos = load_data("data/training_pos.txt")
    sentences_neg = load_data("data/training_neg.txt")

    train_sentences = sentences_pos + sentences_neg

    train_labels = [1 for i in range(len(sentences_pos))] + [0 for i in range(len(sentences_neg))]

    sentences_pos = load_data("data/test_pos_public.txt")
    sentences_neg = load_data("data/test_neg_public.txt")
    test_sentences = sentences_pos + sentences_neg
    test_labels = [1 for i in range(len(sentences_pos))] + [0 for i in range(len(sentences_neg))]
    formatted_vocab_list = []
    
    formatted_vocab_list = list(filter(lambda w: not w in s,vocab_list))
    formatted_vocab_list = [porter_stemmer.stem(word) for word in formatted_vocab_list]
    filter_s = set(words.words())
    
    
    # formatted_vocab_list = list(filter(lambda w: not w in filter_s,formatted_vocab_list))
    formatted_vocab_list = list(filter(lambda w: not w in filter_s,vocab_list))
    formatted_vocab_dict = {item: i for i, item in enumerate(formatted_vocab_list)}
    
    
    
    feat_map = feature_extractor(vocab_list, tokenize)
    # feat_map = feature_extractor(formatted_vocab_list, tokenize)

    # You many replace this with a different feature extractor
    word_freq = compute_word_idf(train_sentences, vocab_list) 

    feat_map_tf_idf = tfidf_extractor(vocab_list, tokenize, word_freq)

    # train with GD
    niter = 100
    print("Training using GD for ", niter, "iterations.")
    d = len(vocab_list)
    # d = len(formatted_vocab_list)
    params = np.array([0.0 for i in range(d)])
    classifier1 = classifier_agent(feat_map,params)
    classifier1.train_gd(train_sentences,train_labels,niter,0.001)
    
    # train GD with tf-idf-feature
    classifieridf = classifier_agent(feat_map_tf_idf,params)
    classifieridf.train_gd(train_sentences,train_labels,niter,0.001)




    # train with SGD
    nepoch = 10
    print("Training using SGD for ", nepoch, "data passes.")
    d = len(vocab_list)
    # d = len(formatted_vocab_list)
    params = np.array([0.0 for i in range(d)])
    classifier2 = classifier_agent(feat_map, params)
    classifier2.train_sgd(train_sentences, train_labels, nepoch, 0.001)
    # train SGD with tf-idf-feature
    classifieridfsgd = classifier_agent(feat_map_tf_idf,params)
    classifieridfsgd.train_gd(train_sentences,train_labels,niter,0.001)


    err1 = classifier1.eval_model(test_sentences,test_labels)
    err2 = classifier2.eval_model(test_sentences,test_labels)
    err3 = classifieridf.eval_model(test_sentences,test_labels)
    err4 = classifieridfsgd.eval_model(test_sentences,test_labels)
        
    classifier1.save_params_to_file('gd_wts.npy')
    classifier2.save_params_to_file('sgd_wts.npy')
    classifier2.save_params_to_file('best_model.npy')
    classifieridfsgd.save_params_to_file('idf_model.npy')
    

    print('GD: test err = ', err1,
          'SGD: test err = ', err2,
          'GD with idf: test err', err3,
          'SGD with idf: test err = ', err4)
    
    
    with open('models_metrics.txt', 'w') as f:
        f.write(f'GD: test err = {err1}\n')
        f.write(f'SGD: test err = {err2}\n')
        f.write(f'GD with idf: test err = {err3}\n')
        f.write(f'SGD with idf : test err = {err4}\n')
         
        
        
        
    


if __name__ == "__main__":
    main()