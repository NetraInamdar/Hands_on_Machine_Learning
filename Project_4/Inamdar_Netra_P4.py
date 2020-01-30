
"""
Created on Fri Sep  6 22:02:24 2019

@author: Netra Inamdar
"""
import matplotlib.pyplot as plt
import numpy as np
import random,math

spam=0
ham=0
counted={}

def cleantext(text):
    text=text.lower()
    text=text.strip()
    for letters in text:
        if letters in """[]!.,"-!-@;':#$%^&*()+/?""":
            text=text.replace(letters," ")
            
    return text

def countwords (words_copy, is_spam , counted):
    for each_word in words_copy:
        if each_word in counted:
            if is_spam == 1:
                counted[each_word][1]=counted[each_word][1]+1
            else:
                counted[each_word][0]=counted[each_word][0]+1
        else:
            if is_spam==1:
                counted[each_word] = [0,1]
            else:
                counted[each_word] = [1,0]
    return counted

def make_percent_list(k, theCount, spams,hams):
    for each_key in theCount:
        theCount[each_key][0] = np.round((theCount[each_key][0] +k)/ (2*k+hams),3)
        theCount[each_key][1] = np.round((theCount[each_key][1] +k)/ (2*k+spams),3)
    return theCount

stop_words=[]
train_set=input("Enter name of training file with extension:")
words_set=input("Enter name of Stop Words file with extension:")

fin=open(words_set,"r")  # check if filename is valid and present in directory
textline=fin.readline()

while textline!="":
    textline = textline.split("\n")
    stop_words.append(textline[0])
    textline=fin.readline()

#print(stop_words)
#data[k,j] = float(t[j]) # store entire file data in data array

  
#train_set='GEASTrain.txt'
fin=open(train_set,"r")  # check if filename is valid and present in directory
textline=fin.readline()
while textline!="":
    is_spam=int(textline[:1])
    if is_spam==1:
        spam=spam+1
    else:
        ham=ham+1
    textline=cleantext(textline[1:])
    words=textline.split()
    words_copy=set(word for word in words if word not in stop_words)
    counted=countwords(words_copy,is_spam,counted)
    textline=fin.readline()
#print(counted)
#print(len(counted))
vocab = (make_percent_list (1, counted, spam, ham))
#print(vocab)
#print(len(vocab))
fin.close()

#stop_words=input("Enter name of file with stop words with extension:")    
#fin=open(stop_words,"r")  # check if filename is valid and present in directory


test_spam=0
test_ham=0

test_set=input("Enter name of test set file with extension:") #'GEASTest.txt'
fin=open(test_set,"r")  # check if filename is valid and present in directory
textline=fin.readline()
while textline!="":
    is_test_spam=int(textline[:1])
    if is_test_spam==1:
        test_spam=test_spam+1
    else:
        test_ham=test_ham+1
    textline=fin.readline()

print('Total spam emails in test set:',test_spam)
print('Total ham emails in test set:',test_ham)
test_spam=test_spam/(test_spam+test_ham)
test_ham=test_ham/(test_spam+test_ham)

ham_count=0
spam_count=0
tp_count=0
fp_count=0
tn_count=0
fn_count=0

fin=open(test_set,"r")  # check if filename is valid and present in directory
textline=fin.readline()
while textline!="":
    is_spam=int(textline[:1])
    textline=cleantext(textline[1:])
    words=textline.split()
    words=set(words)
    #print(words)
    p_spam=1
    p_ham=1
    for key in vocab.keys():
        if key in words:
            p_spam*=vocab[key][1]
            p_ham*=vocab[key][0]
        else:
            p_spam*=(1-vocab[key][1])
            p_ham*=(1-vocab[key][0])
    num=p_spam*test_spam
    denom=(p_spam*test_spam)+(p_ham*test_ham)
    total_prob=num/denom
    if (total_prob>=0.5):
        spam_count+=1
        if is_spam==1:
            tp_count+=1
        elif is_spam==0:
            fp_count+=1
            
    else:
        ham_count+=1
        if is_spam==0:
            tn_count+=1
        elif is_spam==1:
            fn_count+=1
    textline=fin.readline()

#print(ham_count)
#print(spam_count)
print('True positives:',tp_count)
print('False positives:',fp_count)
print('True negatives:',tn_count)
print('False negatives:',fn_count)

accuracy=(tp_count+tn_count)/(tp_count+tn_count+fp_count+fn_count)
precision=(tp_count)/(tp_count+fp_count)
recall=(tp_count)/(tp_count+fn_count)
f1_denom=(1/precision)+(1/recall)
F1=2*(1/f1_denom)

print('Accuracy:',np.round(accuracy,3))
print('Precision:',np.round(precision,3))
print('Recall:',np.round(recall,3))
print('F1 value:',np.round(F1,3))