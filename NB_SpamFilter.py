# -*- coding: utf-8 -*-
"

# Bayesian Network & Naive Bayes Text Classification, Transfer learning for SMS classification

# Original Author: X. Zhu  

# Some codes were adopted from following resources project.
# * Code Credit: https://github.com/hmahajan99/Text-Classification
# * Code Credit: https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt
# * Stopword_Set: https://www.kaggle.com/datasets/rowhitswami/stopwords?resource=download
# * Fractional Features in BoG: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
"""

import os
import csv
import string
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from google.colab import drive
drive.mount('/content/drive')
np.random.seed(55)

path = "/content/drive/My Drive/FAU/Degree-Planning/Ph.D./2023-Spring/Artificial-Intelligence/" 
with open(path + "stopwords.txt", encoding="utf8", errors='ignore') as f:
        # stopwords = f.read()
        stopwords = [line.rstrip() for line in f]

print(stopwords)
print(len(stopwords))

# Create vocabulary (using dictionary)
#topNum specify number of top words. If topNum=0, meaning selecting all words

def count_url(word):
    if any(x in word for x in ["www.","http","https","://",".com",".ru",".net",".co",".ir",".in",".uk","info","biz","us","ca","de","xxx","exposed","xyz","site","online","@"]):
        return int(1)
    else:
        return int(0)

def count_symbols(sentence, symbol_list=None):
    '''
    these are trigger symbols. Do not sort because the order matters for the BoW
    '''
    freq = dict()
    if symbol_list ==  None:
        symbol_list = ["$","@", "%",'?','!',';',':','—','–','-','-','.','£', '₤', '€']
    for symbol in symbol_list:
        freq[symbol] = sentence.count(symbol)
    return( dict(freq) )

def createVocabulary(inDataset, stopwordset,topNum):
    vocab = {}
    # iterate i over every email or SMS
    for i in range(len(inDataset)):
        message = inDataset[i].split()
        for word in message:
            word_new  = re.sub('[\n]', '', word.strip(string.punctuation).lower())
            if (len(word_new)>2)  and (word_new not in stopwordset):  
                if word_new in vocab:
                    vocab[word_new]+=1
                else:
                    vocab[word_new]=1

    # sort the dictionary to focus on most frequent words
    vocab_ordered=sorted(vocab.items(), key=lambda x: x[1],reverse=True)
    #import itertools
    if topNum==0:
        return(dict(vocab_ordered))
    elif topNum<len(vocab):
        V_cut=vocab_ordered[0:topNum]
        V_cut=dict(V_cut)
        return(V_cut)      
    return(dict(vocab_ordered))

# To represent training data as bog of words vector representaton (including counts)
def BoWInstances(inDataset, features):
    # list_of_sentences = inDataset
    # list_of_words     = features (from dictionary)

    # num of symbols, +1 for url counter and +1 for relative capitalization
    additional_features = int(len(count_symbols(inDataset[0])) + 1 + 1)
    inDataset_ = np.zeros((len(inDataset),len(features)+additional_features))

    for i in range(len(inDataset)):
        sentence = inDataset[i]
        message  = inDataset[i].split()
        cap_letter_count, letter_count, url_count = int(0), int(0), int(0)
        symbols_dist_dict = count_symbols(sentence)

        for word in message:
            cap_letter_count  = cap_letter_count  + sum(1 for c in word if c.isupper())
            letter_count      = letter_count + len(word)
            url_count         = url_count + count_url(word)
            word_new  = re.sub('[\n]', '', word.strip(string.punctuation).lower())
            if word_new in features:
                inDataset_[i][features.index(word_new)] += 1
        
        # NEW FEATURES ADDED HERE
        # Add additional integer counts here. E.g., trigger punctuation
        for idx, item in enumerate(symbols_dist_dict):
            symbol = item[0]
            value  = symbols_dist_dict[symbol]
            y_index = int(len(features) + idx)
            inDataset_[i][y_index] = value
        
        # Add fractional counts here. E.g., relative_capitalization_count
        # https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
        relative_capitalization_count = round((cap_letter_count/letter_count), 2) # Could also try binning the percents into ints in a range 0,1,2, ..., 10
        y_index = int(len(features) + len(count_symbols(inDataset[0])) - 1) # 
        inDataset_[i][y_index + 1] = relative_capitalization_count
        inDataset_[i][y_index + 2] = url_count
    # rows = sentence
    # col  = word
    # entry_i_j = count of words in sentence
    return(inDataset_)

# Modified Functions
def CreateLabeledVocabulary(inDataset, inDatasetLabels, stopwordset, topNum):
    '''
    input: Training_Data = inDataset, Training_Labels = inDatasetLabels
    output: dict(vocab_ordered_spam) dict(vocab_ordered_normal) 
    desc:
    '''
    vocab_spam   = {}
    vocab_normal = {}
    for i in range(len(inDataset)):        
        is_spam   = (inDatasetLabels[i] == "spam")
        is_normal = (inDatasetLabels[i] == "normal")

        if is_spam:
            for word in inDataset[i].split():
                word_new  = re.sub('[\n]', '', word.strip(string.punctuation).lower())
                if (len(word_new)>2)  and (word_new not in stopwordset):  
                    if word_new in vocab_spam:
                        vocab_spam[word_new]+=1
                    else:
                        vocab_spam[word_new]=1
        elif is_normal:
            for word in inDataset[i].split():
                word_new  = re.sub('[\n]', '', word.strip(string.punctuation).lower())
                if (len(word_new)>2)  and (word_new not in stopwordset):  
                    if word_new in vocab_normal:
                        vocab_normal[word_new]+=1
                    else:
                        vocab_normal[word_new]=1          
    # sort the dictionary to focus on most frequent words
    vocab_spam_ordered  =sorted(vocab_spam.items(), key=lambda x: x[1],reverse=True)
    vocab_normal_ordered=sorted(vocab_normal.items(), key=lambda x: x[1],reverse=True)
    #import itertools
    if topNum==0:
        return(dict(vocab_spam_ordered), dict(vocab_normal_ordered))
    
    elif topNum < len(vocab_spam) or topNum < len(vocab_normal):
        V_spam_cut = vocab_spam_ordered[0:topNum]
        V_spam_cut = dict(V_spam_cut)
        
        V_normal_cut = vocab_normal_ordered[0:topNum]
        V_normal_cut = dict(V_normal_cut)
        return(V_spam_cut, V_normal_cut)
    
    return(dict(vocab_spam_ordered), dict(vocab_normal_ordered))


def Vocab_Difference(Normal_Dict, Spam_Dict):
    '''
    input: Normal_Dict Spam_Dict
    output: Difference_Dict
    desc: Create vocabulary of the differences between words in Spam/Non-Spam
    key idea: what are spam words that do not show up in non-spam nteractions

    NOTE: difference (subtraction) implies that there can be negative `frequencies`
    '''
    Difference_Dict = Spam_Dict.copy()
    for spam_word in Spam_Dict:
        if spam_word in Normal_Dict:
            Difference_Dict[spam_word] = Normal_Dict[spam_word]-Spam_Dict[spam_word]
    # Analogus to a 'super spam' frequence distribution by word
    # sort the dictionary to focus on most frequent words
    Difference_Dict_ordered  =sorted(Difference_Dict.items(), key=lambda x: x[1],reverse=True)
    return(dict(Difference_Dict_ordered))

def plot(V):
    num_words = [0 for i in range(max(V.values())+1)] 
    freq = [i for i in range(max(V.values())+1)] 
    for key in V:
        num_words[V[key]]+=1
    maxv=max(num_words)+10
    plt.plot(freq,num_words)
    plt.axis([1, 60, 0, maxv])
    plt.xlabel("Frequency")
    plt.ylabel("No of words")
    plt.grid()
    return(plt.show())

# Implementing Multinomial Naive Bayes from scratch
class MultinomialNaiveBayes:
    
    def __init__(self):
        # count is a dictionary which stores several dictionaries corresponding to each news category
        # each value in the subdictionary represents the freq of the key corresponding to that news category 
        self.count = {}
        # classes represents the different news categories
        self.classes = None
    
    def fit(self,X_train,Y_train):
        # This can take some time to complete       
        self.classes = set(Y_train)
        for class_ in self.classes:
            self.count[class_] = {}
            for i in range(len(X_train[0])):
                self.count[class_][i] = 0
            self.count[class_]['total'] = 0
            self.count[class_]['total_points'] = 0
        self.count['total_points'] = len(X_train)
        
        for i in range(len(X_train)):
            for j in range(len(X_train[0])):
                self.count[Y_train[i]][j]+=X_train[i][j]
                self.count[Y_train[i]]['total']+=X_train[i][j]
            self.count[Y_train[i]]['total_points']+=1
    
    def __probability(self,test_point,class_):
        
        log_prob = np.log(self.count[class_]['total_points']) - np.log(self.count['total_points'])
        total_words = len(test_point)
        for i in range(len(test_point)):
            current_word_prob = test_point[i]*(np.log(self.count[class_][i]+1)-np.log(self.count[class_]['total']+total_words))
            log_prob += current_word_prob
        
        return log_prob
    
    
    def __predictSinglePoint(self,test_point):
        
        best_class = None
        best_prob = None
        first_run = True
        
        for class_ in self.classes:
            log_probability_current_class = self.__probability(test_point,class_)
            if (first_run) or (log_probability_current_class > best_prob) :
                best_class = class_
                best_prob = log_probability_current_class
                first_run = False
                
        return best_class
        
  
    def predict(self,X_test):
        # This can take some time to complete
        Y_pred = [] 
        for i in range(len(X_test)):
        # print(i) # Uncomment to see progress
            Y_pred.append( self.__predictSinglePoint(X_test[i]) )
        
        return Y_pred
    
    def score(self,Y_pred,Y_true):
        # returns the mean accuracy
        count = 0
        for i in range(len(Y_pred)):
            if Y_pred[i] == Y_true[i]:
                count+=1
        return count/len(Y_pred)

X = [] # an element of X is represented as (filename, text)
Y = [] # an element of Y represents the category of the corresponding X element

#root_dir='text'

root_dir='/content/drive/My Drive/FAU/Degree-Planning/Ph.D./2023-Spring/Artificial-Intelligence/email'
for category in os.listdir(root_dir):
    for document in os.listdir(root_dir+'/'+category):
        with open(root_dir+'/'+category+'/'+document, "r", encoding="utf8", errors='ignore') as f:
            X.append(f.read())
            Y.append(category)

# print("there are %d messages/files\n %s " % (len(X),X[0:2]))
# print("there are %d labels/files\n %s " % (len(Y),Y[0:2]))

# SMS Data
X_target = []
Y_target = []

with open(path + "sms.csv", encoding="utf8", errors='ignore') as f:
    X_target = [line.rstrip() for line in f]

with open(path + 'labels.csv', encoding="utf8", errors='ignore') as f:
    Y_target = [line.rstrip() for line in f]

## ADDED CODE HERE - IF NEEDED DELETE HERE
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=1)
# len(X_train),len(Y_train),len(X_test),len(Y_test)

# MODIFY EXPERIMENT HERE 1 of 2
V_spam, V_normal = CreateLabeledVocabulary(X_train,Y_train, stopwords, 2000)
Difference_Dict = Vocab_Difference(V_normal, V_spam)
V = Difference_Dict

# MODIFY EXPERIMENT HERE 2 of 2
# V=createVocabulary(X_train,stopwords,2000)
# print(len(V))
# print(V)

plot(V)
# plot(Difference_Dict)

# To represent test data as bag of word vector counts
features = list(V.keys())
X_train_dataset = BoWInstances(X_train,features)
X_test_dataset  = BoWInstances(X_test,features)
# len(X_train_dataset),len(X_test_dataset)

# with np.printoptions(threshold=np.inf):
#     print(X_train_dataset[3])

# Using sklearn's Multinomial Naive Bayes
clf = MultinomialNB()
# Every big matrix has a label AND also, X_train_dataset is a meta-matrix 
clf.fit(X_train_dataset,Y_train)

Y_test_pred = clf.predict(X_test_dataset)
sklearn_score_train = clf.score(X_train_dataset,Y_train)
print("Sklearn's score on training data :",sklearn_score_train)
sklearn_score_test = clf.score(X_test_dataset,Y_test)
print("Sklearn's score on testing data :",sklearn_score_test)
print("Classification report for testing data :-")
print(classification_report(Y_test, Y_test_pred))

# now we are loading a target dataset which is Short messages (SMS). We will try to use previous trained email
# spam filter to classify these short messages.

# import csv
# X_target = []
# Y_target = []

# path = "/content/drive/My Drive/FAU/Degree-Planning/Ph.D./2023-Spring/Artificial-Intelligence/" 

# with open(path + "sms.csv", encoding="utf8", errors='ignore') as f:
#     X_target = [line.rstrip() for line in f]

# with open(path + 'labels.csv', encoding="utf8", errors='ignore') as f:
#     Y_target = [line.rstrip() for line in f]

# # print(X_target[0:5],Y_target[0:5])
# # len(X_target),len(Y_target)

X_target_dataset=BoWInstances(X_target,features)
# Using sklearn's Multinomial Naive Bayes to classify this SMS datasets
Y_test_pred = clf.predict(X_target_dataset)
sklearn_score_test = clf.score(X_target_dataset,Y_target)
print("Sklearn's score on testing data :",sklearn_score_test)
print("Classification report for testing data :-")
print(classification_report(Y_target, Y_test_pred))

X_train_, X_test_, Y_train_, Y_test_ = model_selection.train_test_split(X_target, Y_target, test_size=0.95)

# MODIFY EXPERIMENT HERE 1 of 2
# V_spam, V_normal = CreateLabeledVocabulary(X_train,Y_train, stopwords, 2000)
# Difference_Dict = Vocab_Difference(V_normal, V_spam)
# V_ = Difference_Dict
# features_ = list(V_.keys())

# MODIFY EXPERIMENT HERE 2 of 2
# We have changed features
V_=createVocabulary(X_train_,stopwords,2000)
features_ = list(V_.keys())

X_target_dataset=BoWInstances(X_train_,features_) # dataset is training
X_target_testset=BoWInstances(X_test_,features_)  # testset is testing

clf2_ = MultinomialNB()
clf2_.fit(X_target_dataset,Y_train_)
Y_test_pred_ = clf2_.predict(X_target_testset)
our_score_test = clf2_.score(X_target_testset,Y_test_)
print("Our score on testing data :",our_score_test)
print("Classification report for testing data :-")
print(classification_report(Y_test_, Y_test_pred_))

# # Train three separate Multinomial Naive Bayes classifiers on your training data, each with different hyperparameters, feature sets, or training data.
# clf1 = MultinomialNB(alpha=1.0, fit_prior=True)
# clf2 = MultinomialNB(alpha=0.5, fit_prior=False)
# clf3 = MultinomialNB(alpha=2.0, fit_prior=True)
# clf1.fit(X_train, Y_train_)
# clf2.fit(X_train, Y_train_)
# clf3.fit(X_train, Y_train_)

# # Use the predict method of each classifier to predict the class labels for your test data.
# y_pred1 = clf1.predict(X_test)
# y_pred2 = clf2.predict(X_test)
# y_pred3 = clf3.predict(X_test)

# # Combine the predictions of each classifier by majority voting. 
# # For each test data point, count the number of times each class label appears in the predictions made by the three classifiers. 
# # The class label that appears most frequently is then assigned as the final prediction.
# ensemble_pred = []
# for i in range(len(y_pred1)):
#     votes = {y_pred1[i]: 0, y_pred2[i]: 0, y_pred3[i]: 0}
#     votes[y_pred1[i]] += 1
#     votes[y_pred2[i]] += 1
#     votes[y_pred3[i]] += 1
#     ensemble_pred.append(max(votes, key=votes.get))

# # Finally, evaluate the performance of the ensemble classifier using metrics such as accuracy, precision, recall, and F1-score.
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# accuracy = accuracy_score(Y_test, ensemble_pred)
# precision = precision_score(Y_test, ensemble_pred, average='weighted')
# recall = recall_score(Y_test, ensemble_pred, average='weighted')
# f1 = f1_score(Y_test, ensemble_pred, average='weighted')

# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1-score:", f1)

# # Code adapted from 
# # https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt

# # THIS IS NO LONGER USED

# import json 
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# from collections import defaultdict
# word_freqs = defaultdict(int)

# def vocab_2(inDataset, inDatasetLabels, stopwordset, topNum):
#     for i in range(len(inDataset)):
#         sentence = inDataset[i] #.split()
#         # print(sentence)
#         # return(None)
#     # for sentence in corpus:
#         words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sentence)
#         new_words = [word for word, offset in words_with_offsets]
#         for word in new_words:
#             word_freqs[word] += 1
#     return(word_freqs)

# word_freqs = vocab_2(X_train, Y_train, stopwords, 20000)
# word_freqs = json.loads(json.dumps(word_freqs))
# word_freqs_new = dict({})

# for item in word_freqs.items():
#     word = item[0]
#     freq = item[1]
#     new_word = re.sub('Ġ', '', word)
#     word_freqs_new[new_word] = freq

# print(word_freqs_new)

