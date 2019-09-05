# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
import  pickle as pkl
import tensorflow as tf
from tensorflow.contrib import learn
# Show warnings and errors only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# File paths
tf.flags.DEFINE_string('run_dir', './runs/1561281249', 'Restore the model from this run')
tf.flags.DEFINE_string('checkpoint', 'clf-9000', 'Restore the graph from this checkpoint')
FLAGS = tf.app.flags.FLAGS
# Restore parameters
with open(os.path.join(FLAGS.run_dir,'params.pkl'),'rb') as f:
	params = pkl.load(f,encoding='bytes')
# Restore vocabulary processor
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(os.path.join(FLAGS.run_dir, 'vocab'))
# Load test data
x_raw = '假体多少钱呢'
data = np.array(list(vocab_processor.fit_transform(x_raw)))
# Restor graph
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    # Restore metagraph
    saver = tf.train.import_meta_graph('{}.meta'.format(os.path.join(FLAGS.run_dir,'model',FLAGS.checkpoint)))
    # Restore weights
    saver.restore(sess,os.path.join(FLAGS.run_dir,'model',FLAGS.checkpoint))
    # Get tensor
    input_x = graph.get_tensor_by_name('input_x:0')
    input_y = graph.get_tensor_by_name('input_y:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    predictions = graph.get_tensor_by_name('softmax/predictions:0')
    accuracy = graph.get_tensor_by_name('accuracy/accuracy:0')
    # Test
    result = sess.run(y,feed_dict={x:data})


