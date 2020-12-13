#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import math

#Loading Training and Test data
input_features_c0 = np.loadtxt('Train1.txt')
input_features_c1=np.loadtxt('Train2.txt')

input_features_c0_test=np.loadtxt('Test1.txt')
input_features_c1_test=np.loadtxt('Test2.txt')

#Dividing training dataset into training and validation dataset
input_features_c0_val=input_features_c0[1500:,:]
input_features_c0_train=input_features_c0[0:1500,:]
input_features_c1_val=input_features_c1[1500:,:]
input_features_c1_train=input_features_c1[0:1500,:]

#Building complete training, validation and test input features
input_features_train=np.concatenate((input_features_c0_train, input_features_c1_train), axis=0)
input_features_val=np.concatenate((input_features_c0_val, input_features_c1_val), axis=0)
input_features_test=np.concatenate((input_features_c0_test, input_features_c1_test), axis=0)

#Normalization of training, validation and test data
mean_vector=np.mean(input_features_train, axis = 0)
std_vector=np.std(input_features_train, axis = 0)

input_features_train=(input_features_train-mean_vector)/std_vector
input_features_val=(input_features_val-mean_vector)/std_vector
input_features_test=(input_features_test-mean_vector)/std_vector



#creating target output array for training, validation and test data
target_output_0=np.zeros([1500,1], dtype = int)
target_output_1=np.ones([1500,1], dtype = int)
target_output_train=np.concatenate((target_output_0, target_output_1), axis=0)

target_output_0=np.zeros([500,1], dtype = int)
target_output_1=np.ones([500,1], dtype = int)
target_output_val=np.concatenate((target_output_0, target_output_1), axis=0)

target_output_0=np.zeros([1000,1], dtype = int)
target_output_1=np.ones([1000,1], dtype = int)
target_output_test=np.concatenate((target_output_0, target_output_1), axis=0)


# Sigmoid function :
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivative of sigmoid function :
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))


def training(no_hidden_nodes,lr,j_Loss_train,j_Loss_val,j_Loss_test):
    input_features=input_features_train
    target_output=target_output_train

    weight_hidden = np.random.rand(2,no_hidden_nodes)
    weight_output = np.random.rand(no_hidden_nodes,1)

    min_validation_error=math.inf
    
    # Main logic :
    for epoch in range(50000):
        # Input for hidden layer :
        hlayer_input= np.dot(input_features, weight_hidden)
 
        # Output from hidden layer :
        hlayer_output = sigmoid(hlayer_input)
 
        # Input for output layer :
        olayer_input = np.dot(hlayer_output, weight_output)
 
        # Output from output layer :
        olayer_output = sigmoid(olayer_input)
#========================================================================
        # Phase1
 
        # Calculating Mean Squared Error
        error_out = ((1 / 2) * (np.power((olayer_output - target_output), 2)))
        j_Loss_train.append(error_out.sum()/3000)
 
 
        # Derivatives for phase 1 :
        total_op_error = (olayer_output - target_output)
        activation_olayer_input = sigmoid_der(olayer_input) 
        hlayer_out = hlayer_output
        error_wo = np.dot(hlayer_out.T, total_op_error * activation_olayer_input)
# ========================================================================
         # Phase 2
 
        # Derivatives for phase 2 :
        error_ino = total_op_error * activation_olayer_input
        ino_outh = weight_output
        error_outh = np.dot(error_ino , ino_outh.T)
        outh_inh = sigmoid_der(hlayer_input) 
        inh_wh = input_features
        error_wh = np.dot(inh_wh.T, outh_inh * error_outh)
        
        # Update Weights
        weight_hidden -= lr * error_wh
        weight_output -= lr * error_wo
    
    
    
        #Calculating mean squared error for validation set
        input_hidden_v = np.dot(input_features_val, weight_hidden)
        output_hidden_v = sigmoid(input_hidden_v)
        input_op_v = np.dot(output_hidden_v, weight_output)
        output_op_v = sigmoid(input_op_v)
    
        error_out_v = ((1 / 2) * (np.power((output_op_v - target_output_val), 2)))
        error_out_v=error_out_v.sum()
        j_Loss_val.append(error_out_v/1000)
        
          
        #Calculating mean squared error for test set
        input_hidden_t = np.dot(input_features_test, weight_hidden)
        output_hidden_t = sigmoid(input_hidden_t)
        input_op_t = np.dot(output_hidden_t, weight_output)
        output_op_t = sigmoid(input_op_t)
    
        error_out_t = ((1 / 2) * (np.power((output_op_t - target_output_test), 2)))
        error_out_t=error_out_t.sum()
        j_Loss_test.append(error_out_t/2000)
        
        
        if(min_validation_error<=error_out_v):
                break
        else:
            min_validation_error=error_out_v
    
        
    #Calculating classification accuracy
    input_hidden_t = np.dot(input_features_test, weight_hidden)
    output_hidden_t = sigmoid(input_hidden_t)
    input_op_t = np.dot(output_hidden_t, weight_output)
    output_op_t = sigmoid(input_op_t)

    output_final=np.round(output_op_t).astype(int)
    return np.sum(output_final == target_output_test)/target_output_test.shape[0]
    
    
    
if __name__ == '__main__':
    
    for nh in [2,4,6,8,10]:
        plt.clf()
        avg=0
        print("Accuracy at number of hidden nodes = "+str(nh)+" for 10 iterations are as below:")
        for i in range(10):
            j_Loss_train=[]
            j_Loss_val=[]
            j_Loss_test=[]
            count=0
            acc=training(nh,0.0001,j_Loss_train,j_Loss_val,j_Loss_test)
            avg+=acc
            print("Accuracy at Iteration "+str(i)+" = "+str(acc))
        print("Average Accuracy at number of hidden nodes = "+str(nh)+" is "+str(avg/10)+"\n")
        #fig, ax = plt.subplots()
        plt.plot(j_Loss_train,"-b",label="Training Data")
        plt.plot(j_Loss_val,"-r",label="Validation Data")
        plt.plot(j_Loss_test,"-g",label="Testing Data")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.legend(loc=0, fontsize=15)
        plt.margins(0.02)
        plt.xlabel('Number of epochs')
        plt.ylabel('Average Loss')
        plt.title('Learning curve for MLP 2-'+str(nh)+"-1");
        plt.show()
        
        
        
    


# In[ ]:




