#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###### CPSC 6820 PROJECT 2: Regression ######
###### Author: Netra Inamdar ######
###### Created Date: 9/25/19 ######

import matplotlib.pyplot as plt
import numpy as np
import random
       
'''
filename='GPAData.txt' 
fin=open(filename,"r")
rows = int(fin.readline()) # read no. of rows
data = np.zeros([rows, 3])
test_set = np.zeros([90, 3])
train_set= np.zeros([210,3])
 
for k in range(rows):
    aString= fin.readline()
    t = aString.split("\t")
    for j in range(3):
        data[k,j] = float(t[j]) # store entire file data in data array
fin.close()

# Min max normalization:
study_arr=[]
beer_arr=[]
gpa_arr=[]
for k in range(rows):
    study_arr.append(data[k,0]) # studying minutes values
    beer_arr.append(data[k,1]) #beer per week values
    gpa_arr.append(data[k,2]) #beer per week values

study_min=np.min(study_arr) 
study_max=np.max(study_arr) 
beer_min=np.min(beer_arr) 
beer_max=np.max(beer_arr) 
gpa_min=np.min(gpa_arr) 
gpa_max=np.max(gpa_arr) 

for dataset in data:
    dataset[0]=((dataset[0]-study_min)/(study_max-study_min))*4
    dataset[1]=((dataset[1]-beer_min)/(beer_max-beer_min))*4
    dataset[2]=((dataset[2]-gpa_min)/(gpa_max-gpa_min))*4

random.shuffle(data)
test_set_ind=np.random.choice(rows, 90, replace=False) # select 90 rows 
test_set=data[test_set_ind] # test dataset with 90 records
        
train_set_ind=[x for x in range(rows) if x not in test_set_ind]
train_set=data[train_set_ind] # train set with 210 records    

# initial values of weights and learning rate
w0=5
w1=1
w2=1
w3=2
w4=3
w5=3
alpha=0.001
x0=x1=x2=0

m=210 # Training dataset size
h_sum=0
for i in range(m):
    x0=1
    x1=train_set[i,0] 
    x2=train_set[i,1]
    h=w0*x0+w1*x1+w2*x2+w3*x1*x2+w4*x1*x1+w5*x2*x2 # h calculation
    h_sum+=(h-train_set[i,2])**2 # calculate summation over m records

j_cost_fun=h_sum/(2*m) # initial cost function
#print('initial j is:',j_cost_fun) # Init J value: 2822.33

for i in range(40): # no of iterations selected: 40
    new_error=0
    for j in range(m):
        x0=1
        x1=train_set[j,0]
        x2=train_set[j,1]
        h=w0*x0+w1*x1+w2*x2+w3*x1*x2+w4*x1*x1+w5*x2*x2
        new_error+=(h-train_set[j,2])**2 # new error calculation
    
    temp0_cost=0
    temp1_cost=0
    temp2_cost=0
    temp3_cost=0
    temp4_cost=0
    temp5_cost=0
    for j in range(m):
        x0=1
        x1=train_set[j,0]
        x2=train_set[j,1]
        temp0_cost+= (w0*x0+w1*x1+w2*x2+w3*x1*x2+w4*x1*x1+w5*x2*x2 - train_set[j,2])
        temp1_cost+= (w0*x0+w1*x1+w2*x2+w3*x1*x2+w4*x1*x1+w5*x2*x2 - train_set[j,2])*x1
        temp2_cost+= (w0*x0+w1*x1+w2*x2+w3*x1*x2+w4*x1*x1+w5*x2*x2 - train_set[j,2])*x2
        temp3_cost+= (w0*x0+w1*x1+w2*x2+w3*x1*x2+w4*x1*x1+w5*x2*x2 - train_set[j,2])*x1*x2
        temp4_cost+= (w0*x0+w1*x1+w2*x2+w3*x1*x2+w4*x1*x1+w5*x2*x2 - train_set[j,2])*2*x1
        temp5_cost+= (w0*x0+w1*x1+w2*x2+w3*x1*x2+w4*x1*x1+w5*x2*x2 - train_set[j,2])*2*x2
        
    temp0=w0-(alpha*(1/m)*temp0_cost)
    temp1=w1-(alpha*(1/m)*temp1_cost)
    temp2=w2-(alpha*(1/m)*temp2_cost)
    temp3=w3-(alpha*(1/m)*temp3_cost)
    temp4=w4-(alpha*(1/m)*temp4_cost)
    temp5=w5-(alpha*(1/m)*temp5_cost)

    w0=temp0
    w1=temp1
    w2=temp2
    w3=temp3
    w4=temp4
    w5=temp5
    
    #plt.scatter(i,new_error/(2*m))  # Plot J vs no of iterations
    #plt.xlabel('no. of iterations')
    #plt.ylabel('J (training error)')
    #plt.title('J vs iterations')
    
#print('final j on training set is:', new_error/(2*m))
#print('w0 to w5:',w0,w1,w2,w3,w4,w5) # final values of w0 to w5

m=90 # test dataset size
new_error=0
for j in range(m):
        x0=1
        x1=test_set[j,0]
        x2=test_set[j,1]
        h=w0*x0+w1*x1+w2*x2+w3*x1*x2+w4*x1*x1+w5*x2*x2
        h=abs(h/4) # get values in range 0 to 4 for GPA
        new_error+=(h-test_set[j,2])**2
        
#print('final j on test set is:', new_error/(2*m))
'''

# Final weights used for user input and prediction:
w0: 4.445
w1: -0.309
w2: -0.297
w3: -0.941
w4: 0.381
w5: 0.404
test_val = np.zeros(2)
test_val[0]=input("Enter minutes of studying/week (Enter 0 to exit): ")
test_val[1]=input("Enter ounces of beer/week (Enter 0 to exit): ")

while (test_val[0]!=0 or test_val[1]!=0):
    test_val[0]=((test_val[0]-study_min)/(study_max-study_min))*4
    test_val[1]=((test_val[1]-beer_min)/(beer_max-beer_min))*4

    x0=1
    x1=test_val[0]
    x2=test_val[1]
    h=w0*x0+w1*x1+w2*x2+w3*x1*x2+w4*x1*x1+w5*x2*x2
    h=abs(h%4)
    print('GPA:',str(round(h,2)))
    
    test_val[0]=input("Enter minutes of studying/week: ") # enter next 2 values
    test_val[1]=input("Enter ounces of beer/week: ")
    
print("Program ends here.") # ends when both values entered are 0, 0

