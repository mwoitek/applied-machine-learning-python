# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.8.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Assignment 3 - Evaluation
#
# In this assignment you will train several models and evaluate how effectively
# they predict instances of fraud using data based on [this dataset from
# Kaggle](https://www.kaggle.com/dalpozz/creditcardfraud).
#  
# Each row in `fraud_data.csv` corresponds to a credit card transaction.
# Features include confidential variables `V1` through `V28` as well as
# `Amount` which is the amount of the transaction.
#  
# The target is stored in the `class` column, where a value of 1 corresponds to
# an instance of fraud and 0 corresponds to an instance of not fraud.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC


# %matplotlib notebook

# %% [markdown]
# ## Question 1
#
# Import the data from `fraud_data.csv`. What percentage of the observations in
# the dataset are instances of fraud?
#
# *This function should return a float between 0 and 1.*

# %%
def answer_one():

    # Read the csv file, and create a DataFrame:
    df = pd.read_csv('fraud_data.csv')

    # Create a Series containing the targets:
    target = df.iloc[:, -1]

    # Compute the number of instances of fraud:
    num_fraud = target[target == 1].size

    # Compute the total number of observations:
    num_obs = target.size

    # Compute the ratio between num_fraud and num_obs:
    ratio = num_fraud / num_obs

    return ratio


answer_one()

# %%
# Read the csv file, and create a DataFrame:
df = pd.read_csv('fraud_data.csv')

# Extract the features and targets from this DataFrame:
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split this data into training and test sets.
# Use X_train, X_test, y_train and y_test for all of the following questions.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %% [markdown]
# ## Question 2
#
# Using `X_train`, `X_test`, `y_train` and `y_test` (as defined above), train
# a dummy classifier that classifies everything as the majority class of the
# training data. What is the accuracy of this classifier? What is the recall?
#
# *This function should a return a tuple with two floats, i.e.,
# `(accuracy score, recall score)`.*

# %%
def answer_two():

    # Create a DummyClassifier object:
    clf = DummyClassifier(strategy='most_frequent')

    # Fit this classifier using the training data:
    clf.fit(X_train, y_train)

    # Compute the accuracy of this classifier using the test data:
    accuracy = clf.score(X_test, y_test)

    # Get the predictions for the test data, and compute the recall:
    y_test_pred = clf.predict(X_test)
    recall = recall_score(y_test, y_test_pred)

    return (accuracy, recall)


answer_two()

# %% [markdown]
# ## Question 3
#
# Using `X_train`, `X_test`, `y_train` and `y_test` (as defined above), train a
# SVC classifier using the default parameters. What is the accuracy, recall,
# and precision of this classifier?
#
# *This function should a return a tuple with three floats, i.e.,
# `(accuracy score, recall score, precision score)`.*

# %%
def answer_three():

    # Create an SVC object:
    clf = SVC(random_state=0)

    # Fit this classifier using the training data:
    clf.fit(X_train, y_train)

    # Compute the accuracy using the test data:
    accuracy = clf.score(X_test, y_test)

    # Get the predictions for the test data:
    y_test_pred = clf.predict(X_test)

    # Compute the recall:
    recall = recall_score(y_test, y_test_pred)

    # Compute the precision:
    precision = precision_score(y_test, y_test_pred)

    return (accuracy, recall, precision)


answer_three()

# %% [markdown]
# ## Question 4
#
# Using the SVC classifier with parameters `{'C': 1e9, 'gamma': 1e-07}`, what
# is the confusion matrix when using a threshold of -220 on the decision
# function? Use `X_test` and `y_test`.
#
# *This function should return a confusion matrix, a 2x2 numpy array with 4
# integers.*

# %%
def answer_four():

    #

    return 0


answer_four()

# %% [markdown]
# ### Question 5
#
# Train a logistic regression classifier with default parameters using X_train and y_train.
#
# For the logistic regression classifier, create a precision recall curve and a ROC curve using y_test and the probability estimates for X_test (probability it is fraud).
#
# Looking at the precision recall curve, what is the recall when the precision is `0.75`?
#
# Looking at the ROC curve, what is the true positive rate when the false positive rate is `0.16`?
#
# *This function should return a tuple with two floats, i.e., `(recall, true positive rate)`.*

# %%
def answer_five():

    # Your code here

    return # Return your answer


# %% [markdown]
# ### Question 6
#
# Perform a grid search over the parameters listed below for a Logistic Regression classifier, using recall for scoring and the default 3-fold cross validation.
#
# `'penalty': ['l1', 'l2']`
#
# `'C':[0.01, 0.1, 1, 10, 100]`
#
# From `.cv_results_`, create an array of the mean test scores of each parameter combination. i.e.
#
# |      	| `l1` 	| `l2` 	|
# |:----:	|----	|----	|
# | **`0.01`** 	|    ?	|   ? 	|
# | **`0.1`**  	|    ?	|   ? 	|
# | **`1`**    	|    ?	|   ? 	|
# | **`10`**   	|    ?	|   ? 	|
# | **`100`**   	|    ?	|   ? 	|
#
# *This function should return a 5 by 2 numpy array with 10 floats.*
#
# *Note: do not return a DataFrame, just the values denoted by '?' above in a numpy array. You might need to reshape your raw result to meet the format we are looking for.*

# %%
def answer_six():

    # Your code here

    return # Return your answer


# %%
# Use the following function to help visualize results from the grid search
def GridSearch_Heatmap(scores):
    plt.figure()
    sns.heatmap(scores.reshape(5,2), xticklabels=['l1','l2'], yticklabels=[0.01, 0.1, 1, 10, 100])
    plt.yticks(rotation=0);

#GridSearch_Heatmap(answer_six())
