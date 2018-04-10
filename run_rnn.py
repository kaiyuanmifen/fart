import tensorflow as tf
import numpy as np
import feat_extract
from tensorflow.contrib import rnn
from feat_extract import *
import sys
import pickle
import os


def unison_shuffled_copies(a, b):
  assert len(a) == len(b)
  p = np.random.permutation(len(a))
  return a[p], b[p]

#define constants
#unrolled through 28 time steps
time_steps=10
#hidden LSTM units
num_units=128
#rows of 28 pixels
n_input=40
#learning rate for adam
learning_rate=0.001
#mnist is meant to be classified in 10 classes(0-9).
n_classes=2
#size of batch
batch_size=32

# Get features

POS_EXAMPLES_FILE = './data/pos_examples.pickle'
NEG_EXAMPLES_FILE = './data/neg_examples.pickle'
if os.path.exists(POS_EXAMPLES_FILE):
  with open(POS_EXAMPLES_FILE, 'rb') as fin:
    pos_examples = pickle.load(fin)
else:
  pos_examples = extractFeature('./data/fart_clean1_mono.wav').T
  with open(POS_EXAMPLES_FILE, 'wb') as fout:
    pickle.dump(pos_examples, fout)

if os.path.exists(NEG_EXAMPLES_FILE):
  with open(NEG_EXAMPLES_FILE, 'rb') as fin:
    neg_examples = pickle.load(fin)
else:
  neg_examples_1 = extractFeature('./data/negative_sample_party_mono.wav').T
  neg_examples_2 = extractFeature('./data/negative_sample_voice_mono.wav').T
  neg_examples = np.concatenate([neg_examples_1, neg_examples_2], axis=0)
  with open(NEG_EXAMPLES_FILE, 'wb') as fout:
    pickle.dump(neg_examples, fout)

# Split train/test
np.random.shuffle(pos_examples)
np.random.shuffle(neg_examples)

train_size = 100000
test_size = 5000
train_size_pos = 30000
train_size_neg = train_size - train_size_pos
train_feats = np.ndarray([0, 40])
train_feats = np.concatenate([train_feats, pos_examples[:train_size_pos]], axis=0)
train_feats = np.concatenate([train_feats, neg_examples[:train_size_neg]], axis=0)
test_feats = np.ndarray([0, 40])
test_feats = np.concatenate([test_feats, pos_examples[train_size_pos:]])
test_feats = np.concatenate([test_feats, neg_examples[train_size_neg: train_size_neg + test_size - test_feats.shape[0]:]])

pos_labels = np.array([1] * train_size_pos).astype(np.float32)
neg_labels = np.array([0] * train_size_neg).astype(np.float32)
train_labels = np.concatenate([pos_labels, neg_labels])

# train_feats, train_labels = unison_shuffled_copies(train_feats, train_labels)

# Function that generates next batch
CURSOR = 0
def nextBatch():
  global CURSOR
  batch_feats = np.ndarray([batch_size, time_steps, n_input])
  # batch_labels = np.ndarray([batch_size, time_steps])
  batch_labels = np.ndarray([batch_size, n_classes])
  # batch_labels = train_labels[CURR_BATCH:CURR_BATCH + time_steps * batch_size]
  samples_per_timestep = batch_size * time_steps

  for batch_idx in range(batch_size):
    for t in range(time_steps):
      batch_feats[batch_idx, t, :] = train_feats[CURSOR + t * batch_size + batch_idx]
      batch_labels[batch_idx, :] = train_labels[CURSOR + t * batch_size + batch_idx]
  CURSOR += samples_per_timestep
  return batch_feats, batch_labels

a_feats, a_labels = nextBatch()
b_feats, b_labels = nextBatch()

# sys.exit(0)

graph = tf.Graph()
with graph.as_default():
  out_weights = tf.Variable(tf.random_normal([num_units, n_classes]))
  out_bias = tf.Variable(tf.random_normal([n_classes]))
  x = tf.placeholder("float", [None, time_steps,n_input])
  y = tf.placeholder("float", [None, n_classes])
  input = tf.unstack(x, time_steps, 1)

  lstm_layer = rnn.BasicLSTMCell(num_units, forget_bias=1)
  outputs, _ = rnn.static_rnn(lstm_layer, input, dtype="float32")
  # state = tf.Variable(tf.zeros([batch_size, n_input]))
  # outputs = []
  # for input_ in input:
  #   output, state = lstm_layer(input_, state)
  #   outputs.append(output)


  prediction = tf.matmul(outputs[-1], out_weights) + out_bias

  #loss_function
  loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
  #optimization
  opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

  #model evaluation
  correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
  accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# init = tf.global_variables_initializer()
num_iters = 30
with tf.Session(graph=graph) as sess:
  # sess.run(init)
  iter = 1
  while iter < num_iters:
    # batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
    batch_x, batch_y = nextBatch()
    # batch_x = batch_x.reshape((time_steps, batch_size, n_input))
    print(batch_x.shape)
    print(batch_y.shape)
    sess.run(opt, feed_dict={x: batch_x, y: batch_y})

    if iter % 10 == 0:
      acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
      los = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
      print("For iter ", iter)
      print("Accuracy ", acc)
      print("Loss ", los)
      print("__________________")

    iter = iter + 1