#!/usr/bin/env python

import tensorflow as tf


class BasicRNN:
  def weight_variable(self, shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

  def bias_variable(self, shape):
    initial = tf.zeros(shape)
    return tf.Variable(initial)

  def __init__(self, n_in, maxlen, n_hidden, n_out):
    self.name = 'BasicRNN'
    self.n_in = n_in
    self.maxlen = maxlen
    self.n_hidden = n_hidden
    self.n_out = n_out
    self.W = self.weight_variable([self.n_hidden, self.n_out])
    self.b = self.bias_variable([self.n_out])

  def inference(self, x, batch_size):
    cell = tf.contrib.rnn.BasicRNNCell(self.n_hidden)
    initial_state = cell.zero_state(batch_size, tf.float32)

    state = initial_state
    outputs = []
    with tf.variable_scope(self.name):
      for t in range(self.maxlen):
        if t > 0:
          tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(x[:, t, :], state)
        outputs.append(cell_output)

      output = outputs[-1]

    y = tf.nn.softmax(tf.matmul(output, self.W) + self.b)
    return y

  def loss(self, y, t):
    return tf.reduce_mean(-tf.reduce_sum(t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), reduction_indices=[1]))

  def training(self, loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_step = optimizer.minimize(loss)
    return train_step

  def accuracy(self, y, t):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, axis=1), tf.argmax(t, axis=1)), dtype=tf.float32))

  def summary(self, l, acc):
    with tf.name_scope(self.name):
      tf.summary.scalar('loss', l)
      tf.summary.scalar('accuracy', acc)
      tf.summary.histogram('weight', self.W)
      tf.summary.histogram('bias', self.b)
    return tf.summary.merge_all()


class LSTM(BasicRNN):
  '''
  RNN with long-short term memory
  '''

  def __init__(self, n_in, maxlen, n_hidden, n_out):
    super(LSTM, self).__init__(n_in, maxlen, n_hidden, n_out)
    self.name = 'LSTM'

  def inference(self, x, batch_size):
    cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
    initial_state = cell.zero_state(batch_size, tf.float32)

    state = initial_state
    outputs = []
    with tf.variable_scope('RNN'):
      for t in range(self.maxlen):
        if t > 0:
          tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(x[:, t, :], state)
        outputs.append(cell_output)

    output = outputs[-1]

    y = tf.nn.softmax(tf.matmul(output, self.W) + self.b)
    return y


class LayerNormBasicLSTM(BasicRNN):
  '''
  RNN with LSTM which is added layer normalization and recurrent dropout
  '''

  def __init__(self, n_in, maxlen, n_hidden, n_out):
    super(LayerNormBasicLSTM, self).__init__(n_in, maxlen, n_hidden, n_out)
    self.name = 'LayerNormLSTM'

  def inference(self, x, batch_size):
    cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.n_hidden)
    initial_state = cell.zero_state(batch_size, tf.float32)

    state = initial_state
    outputs = []
    with tf.variable_scope('RNN'):
      for t in range(self.maxlen):
        if t > 0:
          tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(x[:, t, :], state)
        outputs.append(cell_output)

    output = outputs[-1]

    y = tf.nn.softmax(tf.matmul(output, self.W) + self.b)
    return y


class GRU(BasicRNN):
  '''
  RNN with gated recurrent unit
  '''

  def __init__(self, n_in, maxlen, n_hidden, n_out):
    super(GRU, self).__init__(n_in, maxlen, n_hidden, n_out)
    self.name = 'GRU'

  def inference(self, x, batch_size):
    cell = tf.contrib.rnn.GRUCell(self.n_hidden)
    initial_state = cell.zero_state(batch_size, tf.float32)

    state = initial_state
    outputs = []
    with tf.variable_scope('RNN'):
      for t in range(self.maxlen):
        if t > 0:
          tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(x[:, t, :], state)
        outputs.append(cell_output)

    output = outputs[-1]

    y = tf.nn.softmax(tf.matmul(output, self.W) + self.b)
    return y


class BiRNN(BasicRNN):
  '''
  Bidirectional recurrent neural network
  '''

  def __init__(self, n_in, maxlen, n_hidden, n_out):
    super(BiRNN, self).__init__(n_in, maxlen, n_hidden, n_out)
    self.name = 'BiRNN'
    self.W = self.weight_variable([self.n_hidden * 2, self.n_out])
    self.b = self.bias_variable([self.n_out])

  def inference(self, x, batch_size):
    # Bidirectional RNN can receive reshaped batch
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, self.n_in])
    x = tf.split(x, self.maxlen, 0)

    cell_forward = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
    cell_backward = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)

    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(cell_forward, cell_backward, x, dtype=tf.float32)

    output = outputs[-1]

    y = tf.nn.softmax(tf.matmul(output, self.W) + self.b)
    return y
