import numpy as np
import tensorflow as tf
import random

from donut import Donut
from tensorflow import keras as K
from tfsnippet.modules import Sequential
from donut import complete_timestamp, standardize_kpi
from donut import DonutTrainer, DonutPredictor
from tfsnippet.utils import get_variables_as_dict, VariableSaver

# Read the raw data.
rand_values = []
for i in range (10000):
    rand_values.append(random.randrange(1,101))

ts = []
for i  in range(0,10000):
  ts.append(i + 1)

timestamp, values, labels = [ts,rand_values,[]]
# If there is no label, simply use all zeros.
labels = np.zeros_like(values, dtype=np.int32)

# Complete the timestamp, and obtain the missing point indicators.
timestamp, missing, (values, labels) = \
    complete_timestamp(timestamp, (values, labels))

# Split the training and testing data.
test_portion = 0.3
test_n = int(len(values) * test_portion)
train_values, test_values = values[:-test_n], values[-test_n:]
train_labels, test_labels = labels[:-test_n], labels[-test_n:]
train_missing, test_missing = missing[:-test_n], missing[-test_n:]

save_dir = '/root/Teacher/'

# Standardize the training and testing data.
train_values, mean, std = standardize_kpi(
    train_values, excludes=np.logical_or(train_labels, train_missing))
test_values, _, _ = standardize_kpi(test_values, mean=mean, std=std)

# We build the entire model within the scope of `model_vs`,
# it should hold exactly all the variables of `model`, including
# the variables created by Keras layers.
with tf.Session().as_default():
    with tf.variable_scope('model') as model_vs:
        model = Donut(
            h_for_p_x=Sequential([
                K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                               activation=tf.nn.relu),
                K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                               activation=tf.nn.relu),
            ]),
            h_for_q_z=Sequential([
                K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                               activation=tf.nn.relu),
                K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                               activation=tf.nn.relu),
            ]),
            x_dims=120,
            z_dims=5,
        )

# Remember to get the model variables after the birth of a
# `predictor` or a `trainer`.  The :class:`Donut` instances
# does not build the graph until :meth:`Donut.get_score` or
# :meth:`Donut.get_training_objective` is called, which is
# done in the `predictor` or the `trainer`.

      # save variables to `save_dir`
    trainer = DonutTrainer(model=model, model_vs=model_vs)
    trainer.fit(train_values, train_labels, train_missing, mean, std)

    var_dict = get_variables_as_dict(model_vs)
    saver = VariableSaver(var_dict, save_dir)
    saver.save()

with tf.Session().as_default():
    with tf.variable_scope('model') as model_vs:
        model = Donut(
            h_for_p_x=Sequential([
                K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                               activation=tf.nn.relu),
                K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                               activation=tf.nn.relu),
            ]),
            h_for_q_z=Sequential([
                K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                               activation=tf.nn.relu),
                K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                               activation=tf.nn.relu),
            ]),
            x_dims=120,
            z_dims=5,
        )

    DonutTrainer(model=model, model_vs=model_vs)

    saver = VariableSaver(get_variables_as_dict(model_vs), save_dir)
    saver.restore()

    predictor = DonutPredictor(model)
    test_score = predictor.get_score(test_values, test_missing)
    print(len(test_score), len(test_values))
