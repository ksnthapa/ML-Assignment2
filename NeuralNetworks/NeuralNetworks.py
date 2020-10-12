import time
import mlrose

import pandas as pd
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def plot_learning_curve(estimator, title, X, y, axes = None, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].set_title(title)

    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    # define evaluation procedure
    #cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    axes[0].grid()
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross_validation score")

    axes[0].legend(loc="best")

    # Plot fit_time vs score
    axes[1].grid()
    axes[1].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[1].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[1].set_xlabel("fit_times")
    axes[1].set_ylabel("Score")
    axes[1].set_title("NeuralNetworks model Performance using RHC")

    return plt

# load dataset
data = pd.read_csv("adult.csv", header=0, na_values='?')
data = data.dropna()
f = lambda x:1 if x == '>50K' else 0
income = data['income'].apply(f)
data['income'] = income

#use pandas.get_dummies to convert categorical variable into dummy/indicator variables
df_with_dummies = pd.get_dummies(data=data, columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex'])
y = df_with_dummies.income # Target variable

df_with_dummies = df_with_dummies.drop(columns=['income','native.country', 'age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week'])

#include all the features that was created by get_dummies
features = df_with_dummies.columns.values

X = df_with_dummies[features] # Features

#plot_learning_curve(MLPClassifier(hidden_layer_sizes=(100,),solver="sgd",activation="tanh", alpha=0.05, max_iter=1000, learning_rate_init=0.0006, learning_rate="adaptive"),"Learning Curve for MLP Classifier in dataset1", X, y)
#plot_learning_curve(MLPClassifier(),"Learning Curve for MLP Classifier in dataset1", X, y)
#plot_learning_curve(mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu', \
#                                 algorithm = 'random_hill_climb', max_iters = 1000, \
#                                 bias = True, is_classifier = True, learning_rate = 0.1, \
#                                 early_stopping = True, clip_max = 5, max_attempts = 1000, \
#                                 random_state = 3),"Learning Curve for Neural networks using Randomized Hill climbing", X, y)

#plt.savefig('MLP1.png')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
#
# # #classifying the predictors and target variables as X and Y
# # X_train = training_set.iloc[:,0:-1].values
# # Y_train = training_set.iloc[:,-1].values
# # X_test = validation_set.iloc[:,0:-1].values
# # y_test = validation_set.iloc[:,-1].values
# #
# scaler = StandardScaler()
# scaler.fit(X_train)
# #
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
#
# #Initializing the MLPClassifier
# #By default, MLP Classifier uses 'relu' as activation function and 'adam' as cost optimizer or solver
#classifier = MLPClassifier(max_iter=1100)
start =time.time()
#classifier = MLPClassifier(hidden_layer_sizes=(100,),solver="sgd",activation="tanh", alpha=0.05, max_iter=1000, learning_rate_init=0.0006, learning_rate="adaptive")
# Initialize neural network object and fit object
classifier = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu', \
                                 algorithm = 'random_hill_climb', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.1, \
                                 early_stopping = True, clip_max = 5, max_attempts = 1000, \
                                 random_state = 3)
#parameter_space = {
#    'activation': ['tanh', 'relu', 'sigmoid', 'identity'],
#    'learning_rate': ['constant','adaptive'],
#    'learning_rate_init' : [0.01,0.0001,0.00001]
# }
#
#clf = GridSearchCV(classifier, parameter_space, n_jobs=-1, cv=3) #cv 3 is the number of splits for cross-validation
#clf.fit(X_train, y_train)
# # Best parameter set
#print('Best parameters found:\n', clf.best_params_)
#
# # #Fitting the training data to the network
classifier.fit(X_train, y_train)
stop = time.time()
print(f"Training time: {stop - start}s")
# #
# # #Using the trained network to predict
# # #Predicting y for X_val
y_pred = classifier.predict(X_test)
# #
print ("Accuracy score : ", accuracy_score(y_pred, y_test))
# #
# # #Comparing the predictions against the actual observations in y_val
# # cm = confusion_matrix(y_pred, y_test)
# #
# # #Printing the accuracy
# # print("Accuracy of MLPClassifier : ", accuracy(cm))


