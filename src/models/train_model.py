#!/usr/bin/env python

from __future__ import print_function

import tensorflow as tf

from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data

from rnns import BasicRNN
from rnns import LSTM
from rnns import LayerNormBasicLSTM
from rnns import GRU
from rnns import BiRNN

MNIST_WIDTH = 28
MNIST_HEIGHT = 28
BATCH_SIZE = 20

mnist = input_data.read_data_sets('data/MNIST_data/', one_hot=True)

n_out = 10
n_batch = 100

X_train = mnist.train.images
# Convert to sequential data
X_train = X_train.reshape(len(X_train), MNIST_HEIGHT, MNIST_WIDTH)
y_train = mnist.train.labels

X_test = mnist.test.images
X_test = X_test.reshape(len(X_test), MNIST_HEIGHT, MNIST_WIDTH)
y_test = mnist.test.labels

tf.flags.DEFINE_string(
    'model_type', 'basic', "RNN Model type ('basic', 'lstm', 'layernorm_lstm', 'gru', 'birnn') "
)
tf.flags.DEFINE_integer(
    'hidden_size', 120, "Hidden layer size of recurrent unit"
)

FLAGS = tf.flags.FLAGS

def training(model):
    print(X_test.shape)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./graphs/rnn', sess.graph)
        x = tf.placeholder(tf.float32, shape=[None, MNIST_HEIGHT, MNIST_WIDTH], name=model.name)
        t = tf.placeholder(tf.float32, shape=[None, n_out], name=model.name)
        batch_size = tf.placeholder(tf.int32, shape=[])

        y = model.inference(x, batch_size)
        val_loss = model.loss(y, t)
        acc = model.accuracy(y, t)
        train_step = model.training(val_loss)
        summ = model.summary(val_loss, acc)

        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(30):
            X_, y_ = shuffle(X_train, y_train)
            for i in range(n_batch):
                start = i * BATCH_SIZE
                end = start + BATCH_SIZE
                sess.run(train_step, feed_dict={x: X_[start:end],t: y_[start:end], batch_size: BATCH_SIZE})

            l, a, s = sess.run([val_loss, acc, summ], feed_dict={x: X_test, t: y_test, batch_size: len(X_test)})
            print("epoch: {}, loss: {}, accuracy: {}".format(epoch, l, a))
            writer.add_summary(s, global_step=epoch)

    writer.close()


def main(argv):
    if FLAGS.model_type == 'basic':
        model = BasicRNN(MNIST_HEIGHT, MNIST_WIDTH, FLAGS.hidden_size, n_out)
    elif FLAGS.model_type == 'lstm':
        model = LSTM(MNIST_HEIGHT, MNIST_WIDTH, FLAGS.hidden_size, n_out)
    elif FLAGS.model_type == 'layernorm_lstm':
        model = LayerNormBasicLSTM(MNIST_HEIGHT, MNIST_WIDTH, FLAGS.hidden_size, n_out)
    elif FLAGS.model_type == 'gru':
        model = GRU(MNIST_HEIGHT, MNIST_WIDTH, FLAGS.hidden_size, n_out)
    elif FLAGS.model_type == 'birnn':
        model = BiRNN(MNIST_HEIGHT, MNIST_WIDTH, FLAGS.hidden_size, n_out)
    else:
        raise Error("Invalid model type")

    training(model)

if __name__ == '__main__':
    tf.app.run()