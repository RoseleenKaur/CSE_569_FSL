#!/usr/bin/env python
# coding: utf-8

# In[32]:


import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#Loading all datasets with class label
Numpyfile= scipy.io.loadmat('training_data_0.mat') 
training_data_0=(Numpyfile.get('nim0'))
training_data_0=training_data_0.transpose(2,0,1).reshape(5923,-1)
training_data_0=pd.DataFrame(training_data_0)
training_data_0['Class']=0

Numpyfile= scipy.io.loadmat('training_data_1.mat') 
training_data_1=(Numpyfile.get('nim1'))
training_data_1=training_data_1.transpose(2,0,1).reshape(6742,-1)
training_data_1=pd.DataFrame(training_data_1)
training_data_1['Class']=1

Numpyfile= scipy.io.loadmat('testing_data_0.mat') 
testing_data_0=(Numpyfile.get('nim0'))
testing_data_0=testing_data_0.transpose(2,0,1).reshape(980,-1)
testing_data_0=pd.DataFrame(testing_data_0)
testing_data_0['Class']=0

Numpyfile= scipy.io.loadmat('testing_data_1.mat') 
testing_data_1=(Numpyfile.get('nim1'))
testing_data_1=testing_data_1.transpose(2,0,1).reshape(1135,-1)
testing_data_1=pd.DataFrame(testing_data_1)
testing_data_1['Class']=1

#combining training and test data of both classes( 0 and 1)
training_data_all=training_data_0.append(training_data_1, ignore_index = True) 
testing_data_all=testing_data_0.append(testing_data_1, ignore_index = True)



#Task 1. Feature normalization (Data conditioning) of training and test data
mean_vector=np.array(training_data_all.mean())
std_vector=np.array(training_data_all.std())
training_data_all[training_data_all.columns[0:784]]=(training_data_all[training_data_all.columns[0:784]]-mean_vector[0:784])/std_vector[0:784]
testing_data_all[testing_data_all.columns[0:784]]=(testing_data_all[testing_data_all.columns[0:784]]-mean_vector[0:784])/std_vector[0:784]
print('Task 1 results:')
print('Normalized training data samples:')
print(training_data_all.head(5))
print('\n')
print('Normalized testing data samples:')
print(testing_data_all.head(5))
print('\n')



#Task 2. PCA using the training samples. 
cov_mat = np.cov(training_data_all[training_data_all.columns[0:784]].transpose())
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
#index of first 2 principal components
index=eig_vals.argsort()[-2:][::-1]
print('Task 2 results:\n')
print('Variance on PCA 1 (eigen value): ',eig_vals[0])
print('Variance on PCA 2 (eigen value): ',eig_vals[1])
print('\n')
print('PCA 1 (eigen vector): \n',eig_vecs[0])
print('PCA 2 (eigen vector): \n',eig_vecs[1])
print('\n')
print('Variance covered by the first 4 PCA components as percentage of total variance:')
print("""
PCA 1: {0:.2f}% of the variance
PCA 2:  {1:.2f}% of the variance
PCA 3:  {2:.2f}% of the variance
PCA 4:  {3:.2f}% of the variance
""".format(*tuple(eig_vals / np.sum(eig_vals) * 100)))



#Task 3. Dimension reduction using PCA. 
#Projecting training and testing data on first 2 Principal components
proj_training_data_all = pd.DataFrame(np.dot(training_data_all.loc[:,training_data_all.columns[0:784]], eig_vecs[:,index]),
                     columns=['projection_PCA1','projection_PCA2'])
proj_training_data_all['Class'] = training_data_all['Class']
proj_testing_data_all = pd.DataFrame(np.dot(testing_data_all.loc[:,testing_data_all.columns[0:784]], eig_vecs[:,index]),
                     columns=['projection_PCA1','projection_PCA2'])
proj_testing_data_all['Class'] = testing_data_all['Class']

print('\nTask 3 results:\n')
print('Projected training data samples:')
print(proj_training_data_all.head(5))
print('\n')
print('Projected testing data samples:')
print(proj_testing_data_all.head(5))


# Plotting projection1 vs projection2 for training data
plot1 = plt.figure(1)
for key, group in proj_training_data_all.groupby(['Class']):
    plt.plot(group['projection_PCA1'], group['projection_PCA2'],
               label=key, marker='o', linestyle='none')
# Tiding up plot
plt.legend(loc=0, fontsize=15)
plt.margins(0.02)
plt.xlabel('projection_PCA1')
plt.ylabel('projection_PCA2')
plt.title('MNIST training data with principal components');
plt.savefig('MNIST training data with principal components.png')

plot1 = plt.figure(2)
# Plotting projection1 vs projection2 for testing data
for key, group in proj_testing_data_all.groupby(['Class']):
    plt.plot(group['projection_PCA1'], group['projection_PCA2'],
               label=key, marker='o', linestyle='none')
# Tiding up plot
plt.legend(loc=0, fontsize=15)
plt.margins(0.02)
plt.xlabel('projection_PCA1')
plt.ylabel('projection_PCA2')
plt.title('MNIST testing data with principal components');
plt.savefig('MNIST testing data with principal components.png')
plt.show()



#Task 4. Density estimation. 
#parameter estimation for class-0 using its training data
data_0=proj_training_data_all[proj_training_data_all['Class']==0]
mean_0=data_0[data_0.columns[0:2]].mean()
cov_mat_0 = np.cov(data_0[data_0.columns[0:2]].transpose())
print('Task 4 results:\n')
print('Parameters estimation for Class-0: ')
print('Mean              : ',mean_0.to_list())
print('Covariance matrix : ',cov_mat_0)
print('\n\n')
#parameter estimation for class-1 using its training data
data_1=proj_training_data_all[proj_training_data_all['Class']==1]
mean_1=data_1[data_1.columns[0:2]].mean()
cov_mat_1 = np.cov(data_1[data_1.columns[0:2]].transpose())
print('Parameters estimation for Class-1: ')
print('Mean             : ',mean_1.to_list())
print('Covariance matrix: ',cov_mat_1)
print('\n\n')

#Task 5. Bayesian Decision Theory for optimal classification. 
#Bayesian Classification for training data
training_data=proj_training_data_all
pdf_0 = multivariate_normal(mean=mean_0, cov=cov_mat_0)
pdf_1 = multivariate_normal(mean=mean_1, cov=cov_mat_1)
training_data['P(x|class_0)']=pdf_0.pdf(training_data[training_data.columns[0:2]])
training_data['P(x|class_1)']=pdf_1.pdf(training_data[training_data.columns[0:2]])

training_data['Bayesian Classification'] = np.where(training_data['P(x|class_0)']>training_data['P(x|class_1)'], 0, 1)
accuracy_of_training_dset=(training_data[training_data['Bayesian Classification']==training_data['Class']].shape[0]/training_data.shape[0])*100
print('Task 5 results:\n')
print('Accuracy of training dataset: ',accuracy_of_training_dset)

#Bayesian Classification for testing data
testing_data=proj_testing_data_all

testing_data['P(x|class_0)']=pdf_0.pdf(testing_data[testing_data.columns[0:2]])
testing_data['P(x|class_1)']=pdf_1.pdf(testing_data[testing_data.columns[0:2]])

testing_data['Bayesian Classification'] = np.where(testing_data['P(x|class_0)']>testing_data['P(x|class_1)'], 0, 1)
accuracy_of_testing_dset=(testing_data[testing_data['Bayesian Classification']==testing_data['Class']].shape[0]/testing_data.shape[0])*100
print('Accuracy of testing dataset: ',accuracy_of_testing_dset)

