#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 17 Jun 2015 17:51:02 CEST

import logging
logger = logging.getLogger()

import numpy
import bob.learn.linear


def make_labels(X):
  """Helper function that generates a single 1D numpy.ndarray with labels which
  are good targets for stock logistic regression.


  Parameters:

    X (numpy.ndarray): The input data matrix. This must be a numpy.ndarray
      with 3 dimensions or an iterable containing 2 numpy.ndarrays with 2
      dimensions each. Each correspond to the data for one of the two classes,
      every row corresponds to one example of the data set, every column, one
      different feature.


  Returns:

    numpy.ndarray: With a single dimension, containing suitable labels for all
      rows and for all classes defined in X (depth).

  """

  return numpy.hstack([k*numpy.ones(len(X[k]), dtype=int) for k in range(len(X))])


def add_bias(X):
  """Helper function to add a bias column to the input array X


  Parameters:

    X (numpy.ndarray): The input data matrix. This must be a numpy.ndarray
      with 2 dimension wheres every row corresponds to one example of the data
      set, every column, one different feature.


  Returns:

    numpy.ndarray: The same input matrix X with an added (prefix) column of
      ones.

  """

  return numpy.hstack((numpy.ones((len(X),1), dtype=X.dtype), X))


class MultiClassMachine:
  """A class to handle all run-time aspects for Multiclass Log. Regression


  Parameters:

    machines (iterable): An iterable over any number of machines that will be
      stored.

  """


  def __init__(self, machines):
    self.machines = machines


  def __call__(self, X):
    """Spits out the hypothesis for each machine given the data


    Parameters:

      X (numpy.ndarray): The input data matrix. This must be a numpy.ndarray
        with 2 dimensions. Every row corresponds to one example of the data
        set, every column, one different feature.


    Returns:

      numpy.ndarray: A 2D numpy.ndarray with as many entries as rows in the
        input 2D array ``X``, representing g(x), the sigmoidal hypothesis. Each
        column on the output array represents the output of one of the logistic
        regression machines in this

    """

    return numpy.hstack([m(add_bias(X)) for m in self.machines])


  def predict(self, X):
    """Predicts the class of each row of X


    Parameters:

      X (numpy.ndarray): The input data matrix. This must be a numpy.ndarray
        with 3 dimensions or an iterable containing 2 numpy.ndarrays with 2
        dimensions each. Each correspond to the data for one of the two classes,
        every row corresponds to one example of the data set, every column, one
        different feature.


    Returns:

      numpy.ndarray: A 1D numpy.ndarray with as many entries as rows in the
        input 2D array ``X``, representing g(x), the class predictions for the
        current machine.

    """

    return self(X).argmax(axis=1)


class MultiClassTrainer:
  """A class to handle all training aspects for Multiclass Log. Regression


  Parameters:

    regularizer (float): A regularization parameter

  """


  def __init__(self, regularizer=0.0):
    self.regularizer = regularizer


  def train(self, X):
    """
    Trains multiple logistic regression classifiers to handle the multiclass
    problem posed by ``X``

      X (numpy.ndarray): The input data matrix. This must be a numpy.ndarray
        with 3 dimensions or an iterable containing 2 numpy.ndarrays with 2
        dimensions each. Each correspond to the data for one of the input
        classes, every row corresponds to one example of the data set, every
        column, one different feature.


    Returns:

      Machine: A trained multiclass machine.

    """

    _trainer = bob.learn.linear.CGLogRegTrainer(**{'lambda':self.regularizer})

    if len(X) == 2: #trains and returns a single logistic regression classifer

      return _trainer.train(add_bias(X[0]), add_bias(X[1]))

    else: #trains and returns a multi-class logistic regression classifier

      # use one-versus-all strategy
      machines = []
      for k in range(len(X)):
        NC_range = list(range(0,k)) + list(range(k+1,len(X)))
        machines.append(_trainer.train(add_bias(numpy.vstack(X[NC_range])),
            add_bias(X[k])))

      return MultiClassMachine(machines)
