#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# CPSC 6820 PROJECT 1: k Nearest Neighbor
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 22:02:24 2019

@author: Netra Inamdar
"""
import matplotlib.pyplot as plt
import numpy as np
import random

def euclideanDis(val_set,train_set): # calculate euclidean distance 
    distances=np.zeros([len(train_set),2])
    for n in range(len(train_set)):
        dist=np.sqrt((float(val_set[0])-train_set[n,0])**2+                      (float(val_set[1])-train_set[n,1])**2)
        distances[n,0]=dist # distances
        distances[n,1]=(train_set[n,2]) # fish species
    return distances

def kMinValues(dist_array,k): # calculate predictions based on k neighbors
    ans_sum=0
    dist_array=sorted(dist_array,key=lambda x: x[0])
    for i in range(k):
        ans_sum+= dist_array[i][1]  # add predictions for k neighbors
    if ans_sum>=((k+1)/2):
        return 1 # predicted as TigerFish1
    else:
        return 0 # predicted as TigerFish0 
    
def get_cv_accuracy(validation_set,training_set,error_count_dict): # accuracy for K fold CV
    for val_set in validation_set:
        dist_val_set=np.zeros([len(training_set),2])
        dist_val_set=euclideanDis(val_set,training_set) # calculate euclidean distance
        final_ans=np.zeros([48,2])
        for k in range(1,22,2): # varying k from 1 to 21
            final_ans=kMinValues(dist_val_set,k)
            if int(val_set[2]!=final_ans):
                error_count_dict[k]+=1 # update error count for each training & validation set
    #print(error_count_dict.values())
    return error_count_dict
    
def plot_cv_accuracy(error_count_dict): # plot accuracy with different k values
    k_values=[]
    final_error_values=[]
    accuracy_percentage=[]
    for key,val in error_count_dict.items():
        k_values.append(key)
        final_error_values.append(val)
        accuracy_percentage.append(np.round(100-((val/240)*100),3))
    plt.plot(k_values,accuracy_percentage, marker='o', color='b')
    plt.xlabel("Value of k of KNN")
    plt.ylabel("Cross-Validated accuracy")
    plt.title("Avg accuracy for different values of k")
        
filename='FF33.txt' # Given data with 3 columns
fin=open(filename,"r")

rows = int(fin.readline()) # read no. of rows

data = np.zeros([rows, 3])
test_set = np.zeros([60, 3])
val_set1 = np.zeros([48, 3]) 
val_set2 = np.zeros([48, 3])
val_set3 = np.zeros([48, 3])
val_set4 = np.zeros([48, 3])
val_set5 = np.zeros([48, 3])
train_set1 = np.zeros([192, 3])
train_set2 = np.zeros([192, 3])
train_set3 = np.zeros([192, 3])
train_set4 = np.zeros([192, 3])
train_set5 = np.zeros([192, 3])
 
for k in range(rows):
    aString= fin.readline()
    t = aString.split("\t")
    for j in range(3):
        data[k,j] = float(t[j]) # store entire file data in data array
fin.close()

# Standardization:
body_len_arr=[]
fin_len_arr=[]
for k in range(rows):
    body_len_arr.append(data[k,0]) # body length values
    fin_len_arr.append(data[k,1]) #fin length values
    
body_mean=np.mean(body_len_arr) # mean of body length values
body_std=np.std(body_len_arr) # std deviation of body length values
fin_mean=np.mean(fin_len_arr) # mean of fin length values
fin_std=np.std(fin_len_arr) # std deviation of fin length values

for dataset in data:
    dataset[0]=(dataset[0]-body_mean)/body_std
    dataset[1]=(dataset[1]-fin_mean)/fin_std
 
'''
for k in range(rows):
    print(data[k,0],data[k,1])
    if data[k,2] == 0:
        plt.scatter(data[k,0], data[k,1], color = "red",
        marker = "o", label = "TigerFish0")
    else:
        plt.scatter(data[k,0], data[k,1], color = "green",
        marker = "v", label = "TigerFish1")    
    plt.xlabel("This is Body Length (x axis)")
    plt.ylabel("This is Fin Length (y axis)")
    plt.title("Body Length VS Fin Length")
'''

tf1_count=0
tf0_count=0

random.shuffle(data) # shuffle data once
test_set_ind=np.random.choice(rows, 60, replace=False) # select 60 rows 
test_set=data[test_set_ind] # test dataset with 60 records
for t_set in test_set:
    if t_set[2]==1:
        tf1_count+=1 # check how many TigerFish1 species are in test set
    else:
        tf0_count+=1 # check how many TigerFish0 species are in test set
#print('TF1:',tf1_count)
#print('TF0:',tf0_count)
        
train_set_ind=[x for x in range(rows) if x not in test_set_ind]
train_set=data[train_set_ind] # train set with 240 records

val1_ind=np.random.choice(train_set_ind, 48, replace=False)
train1_ind=[x for x in train_set_ind if x not in val1_ind]
val2_ind=np.random.choice(train1_ind, 48, replace=False)
train2_ind=[x for x in train1_ind if x not in val2_ind]
val3_ind=np.random.choice(train2_ind, 48, replace=False)
train3_ind=[x for x in train2_ind if x not in val3_ind]
val4_ind=np.random.choice(train3_ind, 48, replace=False)
train4_ind=[x for x in train3_ind if x not in val4_ind]
val5_ind=np.random.choice(train4_ind, 48, replace=False)
train5_ind=[x for x in train4_ind if x not in val5_ind]

val_set1=data[val1_ind]   # Validation set 1
val_set2=data[val2_ind]   # Validation set 2
val_set3=data[val3_ind]   # Validation set 3
val_set4=data[val4_ind]   # Validation set 4
val_set5=data[val5_ind]   # Validation set 5

train_set2_ind=[x for x in train_set_ind if x not in val2_ind]
train_set3_ind=[x for x in train_set_ind if x not in val3_ind]
train_set4_ind=[x for x in train_set_ind if x not in val4_ind]
train_set5_ind=[x for x in train_set_ind if x not in val5_ind]

train_set1=data[train1_ind]          # Training set 1
train_set2=data[train_set2_ind]      # Training set 2
train_set3=data[train_set3_ind]      # Training set 3
train_set4=data[train_set4_ind]      # Training set 4
train_set5=data[train_set5_ind]      # Training set 5
                   
error_count_dict={1:0,3:0,5:0,7:0,9:0,11:0,13:0,15:0,17:0,19:0,21:0} # error count for 5 fold CV

error_count_dict=get_cv_accuracy(val_set1,train_set1,error_count_dict) #get accuracy for 1st fold
error_count_dict=get_cv_accuracy(val_set2,train_set2,error_count_dict) #get accuracy for 2nd fold
error_count_dict=get_cv_accuracy(val_set3,train_set3,error_count_dict) #get accuracy for 3rd fold
error_count_dict=get_cv_accuracy(val_set4,train_set4,error_count_dict) #get accuracy for 4th fold
error_count_dict=get_cv_accuracy(val_set5,train_set5,error_count_dict) #get accuracy for 5th fold

#print(error_count_dict.values()) # print error counts for all 5 sets
#plot_cv_accuracy(error_count_dict) # plot accuracy vs k values for 5 fold Cross Validation

test_error_count=0
for val_set in test_set:
    dist_val_set=np.zeros([len(train_set),2])
    dist_val_set=euclideanDis(val_set,train_set) # calculate euclidean distance for test set records
    final_ans=np.zeros([60,2])
    
    final_ans=kMinValues(dist_val_set,k=7) # get final answer with k=7
    if (int(val_set[2])!=final_ans):
        #print(val_set, final_ans)
        test_error_count+=1 # calculate error count for test set records
#print('test error count:',test_error_count)  # ranges from 1 to 6

filename=input("Enter name of input file with extension:")
while(filename):
    try:    
        fin=open(filename,"r")  # check if filename is valid and present in directory
        break
    except FileNotFoundError: 
        print("File not found, please check the filename and path again!")
        filename=input("Enter name of input file with extension:")  

rows = int(fin.readline()) # check no. of rows
data = np.zeros([rows, 3])
test_val = np.zeros(2)

for k in range(rows):
    aString= fin.readline()
    t = aString.split("\t")
    for j in range(3):
        data[k,j] = float(t[j]) # store entire dataset in data array
fin.close()

# Standardization:
body_len_arr=[]
fin_len_arr=[]
for k in range(rows):
    body_len_arr.append(data[k,0]) # body length values
    fin_len_arr.append(data[k,1]) #fin length values
    
body_mean=np.mean(body_len_arr) # mean of body length values
body_std=np.std(body_len_arr) # std deviation of body length values
fin_mean=np.mean(fin_len_arr) # mean of fin length values
fin_std=np.std(fin_len_arr) # std deviation of fin length values

for dataset in data:
    dataset[0]=(dataset[0]-body_mean)/body_std
    dataset[1]=(dataset[1]-fin_mean)/fin_std

test_val[0]=input("Enter body length in cms (Enter 0 to exit): ")
test_val[1]=input("Enter dorsal fin length in cms (Enter 0 to exit): ")

while (test_val[0]!=0.0 and test_val[1]!=0.0):
    dist_val_set=np.zeros([len(data),2])
    dist_val_set=euclideanDis(test_val,data) # calculate euclidean distance
    final_ans=np.zeros([1,2])

    final_ans=kMinValues(dist_val_set,k=7) # get the predictions with k=7 value
    if final_ans==0:
        print('TigerFish0') # check prediction value and print species type
    else:
        print('TigerFish1')
    
    test_val[0]=input("Enter next body length in cms: ") # enter next 2 values
    test_val[1]=input("Enter next dorsal fin length in cms: ")
    
print("Program ends here.") # ends when both values entered are 0, 0

