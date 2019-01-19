
# coding: utf-8

# In[53]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[54]:


# step - 1 - Define our data


# In[55]:


# input data - of the form(X value,Y value,Bias term)

X = np.array([
    [-2,4,1],
    [4,1,-1],
    [1,6,-1],
    [2,4,-1],
    [6,2,-1]
])

#Associated output labels - 
y = np.array([-1,-1,1,1,1])


# In[56]:


# plot graph in 2D
for d,sample in enumerate(X):
    # plot the negative samples
    if d<2:
        plt.scatter(sample[0],sample[1],linewidths=2,marker="_")
    # plot the positive samples
    else:
        plt.scatter(sample[0],sample[1],marker="+",linewidths=2)

plt.plot([-2,6],[6,0.5])


# In[57]:


# perform gradient decent to learn the seperating hyperplane

def svm_sgd_plot(X,Y):
    # initialize SVMs weight vector with zeroes(3 values)
    w=np.zeros(len(X[0]))
    #The learning rate
    eta=1
    #How many iterations to train for
    epochs = 100000
    #Store misclassification so we can plot how they change over time
    errors = []
    
    #training part,gradient descent part
    for epoch in range(1,epochs):
        error = 0
        for i ,x in enumerate(X):
            #misclassification  and np.dot is a dot product of two numpy arrays
            if (Y[i]*np.dot(X[i],w)) < 1:
                #misclassifies update for our weights
                w = w + eta * ((X[i] * Y[i]) + (-2 * (1/epoch)*w))
                error = 1
            else:
                #correct classification
                w = w + eta * (-2 * (1/epoch)*w)
        errors.append(error)
        
    #plot the rate of classification errors
    plt.plot(errors, '|')
    plt.ylim(0.5,1.5)
    plt.axes().set_yticklabels([])
    plt.xlabel('Epoch')
    plt.ylabel('Misclassified')
    plt.show()
    return w


# In[58]:


for d, sample in enumerate(X):
    # Plot the negative samples
    if d < 2:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    # Plot the positive samples
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

# Add our test samples
plt.scatter(2,2, s=120, marker='_', linewidths=2, color='yellow')
plt.scatter(4,3, s=120, marker='+', linewidths=2, color='blue')

# Print the hyperplane calculated by svm_sgd()
x2=[w[0],w[1],-w[1],w[0]]
x3=[w[0],w[1],w[1],-w[0]]

x2x3 =np.array([x2,x3])
X,Y,U,V = zip(*x2x3)
ax = plt.gca()
ax.quiver(X,Y,U,V,scale=1, color='blue')


# In[ ]:


w = svm_sgd_plot(X,y)
#they decrease over time! Our SVM is learning the optimal hyperplane

