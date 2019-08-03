#!/usr/bin/env python
# coding: utf-8

# # Predicting Student Admissions with Neural Networks
# 
# 
# 
# 
# ## Loading the data
# 

# In[1]:


# Importing pandas and numpy
import pandas as pd
import numpy as np

# Reading the csv file into a pandas DataFrame
data = pd.read_csv('student_data.csv')

# Printing out the first 10 rows of our data
data[:10]


# ## Plotting the data
# 
# First let's make a plot of our data to see how it looks. In order to have a 2D plot, let's ingore the rank.

# In[2]:


# Importing matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Function to help us plot
def plot_points(data):
    X = np.array(data[["gre","gpa"]])
    y = np.array(data["admit"])
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'red', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'cyan', edgecolor = 'k')
    plt.xlabel('Test (GRE)')
    plt.ylabel('Grades (GPA)')
    
# Plotting the points
plot_points(data)
plt.show()


# Roughly, it looks like the students with high scores in the grades and test passed, while the ones with low scores didn't, but the data is not as nicely separable as we hoped it would. Maybe it would help to take the rank into account? Let's make 4 plots, each one for each rank.

# In[3]:


# Separating the ranks
data_rank1 = data[data["rank"]==1]
data_rank2 = data[data["rank"]==2]
data_rank3 = data[data["rank"]==3]
data_rank4 = data[data["rank"]==4]

# Plotting the graphs
plot_points(data_rank1)
plt.title("Rank 1")
plt.show()
plot_points(data_rank2)
plt.title("Rank 2")
plt.show()
plot_points(data_rank3)
plt.title("Rank 3")
plt.show()
plot_points(data_rank4)
plt.title("Rank 4")
plt.show()


#  we should one-hot encode it.
# 
# ##  One-hot encoding the rank
# Using the `get_dummies` function in pandas in order to one-hot encode the data.
# 
# To drop a column, we use `one_hot_data`[.drop( )]

# In[20]:


# Making dummy variables for rank and concat existing columns
one_hot_data = pd.concat([data, pd.get_dummies(data['rank'], prefix='rank')], axis=1)

# Dropping the previous rank column
one_hot_data = one_hot_data.drop('rank', axis=1)

# Printing the first 10 rows of our data
one_hot_data[:10]


# ##  Scaling the data
#  Our data is skewed, and that makes it hard for a neural network to handle. Let's fit our two features into a range of 0-1, by dividing the grades by 4.0, and the test score by 800.

# In[6]:


# Making a copy of our data
processed_data = one_hot_data[:]

# Scale the columns
processed_data['gre'] = processed_data['gre']/800
processed_data['gpa'] = processed_data['gpa']/4.0
processed_data[:10]

# Printing the first 10 rows of our procesed data
processed_data[:10]


# ## Splitting the data into Training and Testing

# In order to test our algorithm, we'll split the data into a Training and a Testing set. The size of the testing set will be 10% of the total data.

# In[7]:


sample = np.random.choice(processed_data.index, size=int(len(processed_data)*0.9), replace=False)
train_data, test_data = processed_data.iloc[sample], processed_data.drop(sample)

print("Number of training samples is", len(train_data))
print("Number of testing samples is", len(test_data))
print(train_data[:10])
print(test_data[:10])


# ## Splitting the data into features and targets (labels)
# Now, as a final step before the training, we'll split the data into features (X) and targets (y).

# In[8]:


features = train_data.drop('admit', axis=1)
targets = train_data['admit']
features_test = test_data.drop('admit', axis=1)
targets_test = test_data['admit']

print(features[:10])
print(targets[:10])


# ## Training the 2-layer Neural Network
#  writing  some helper functions.

# In[9]:


# Activation (sigmoid) function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))
def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)


# # Backpropagating the error
#  $$ (y-\hat{y}) \sigma'(x) $$

# In[10]:


# Write the error term formula
def error_term_formula(x, y, output):
    return (y - output)*sigmoid_prime(x)


# In[32]:


# Neural Network hyperparameters

epochs = 1000
learnrate = 0.5

# Training function

def train_nn(features, targets, epochs, learnrate):
    
    # Use to same seed to make debugging easier
    np.random.seed(42)

    n_records, n_features = features.shape
    last_loss = None

    # Initializing random weights
    
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        for x, y in zip(features.values, targets):
            
            # Looping through all records, x is the input, y is the target

            # Activation of the output unit
            #   Notice we multiply the inputs and the weights here 
            #   rather than storing h as a separate variable 
            
            output = sigmoid(np.dot(x, weights))

            # The error, the target minus the network output
            
            error = error_formula(y, output)

            # The error term
            
            error_term = error_term_formula(x, y, output)

            # The gradient descent step, the error times the gradient times the inputs
            
            del_w += error_term * x

        # Updating the weights here. The learning rate times the 
        # change in weights, divided by the number of records to average
        
        weights += learnrate * del_w / n_records

        # Printing out the mean square error on the training set
        if e % (epochs / 10) == 0:
            out = sigmoid(np.dot(features, weights))
            loss = np.mean((out - targets) ** 2)
            print("Epoch:", e)
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            print("=========")
    print("Finished training!")
    return weights
    
weights = train_nn(features, targets, epochs, learnrate)


# ## Calculating the Accuracy on the Test Data

# In[45]:


# Calculating accuracy on test data
test_out = sigmoid(np.dot(features_test, weights))
predictions = test_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))

