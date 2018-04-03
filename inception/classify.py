from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import argparse
import sys, os
import tensorflow as tf
from PIL import Image
from tqdm import tqdm


def load_image(filename):
    # Read in the image_data to be classified."""
    return tf.gfile.FastGFile(filename, 'rb').read()


def load_labels(filename):
    # Read in labels, one label per line."""
    return [line.rstrip() for line in tf.gfile.GFile(filename)]


def load_graph(filename):
    # Unpersists graph from file as default graph."""
    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


def run_graph(src, labels, input_layer_name, output_layer_name, num_top_predictions):
    with tf.Session() as sess:
        # Feed the image_data as input to the graph.
        # predictions  will contain a two-dimensional array, where one
        # dimension represents the input image count, and the other has
        # predictions per class
        # with open('submit.csv','w') as outfile:
        for f in os.listdir(src):
            image_data = load_image(os.path.join(src, f))
            softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
            predictions, = sess.run(softmax_tensor, {input_layer_name: image_data})

            print(f)
            for i in range(len(predictions)):
                print(labels[i], ": ", "%.2f" % (predictions[i] * 100), "%")

            print(" ")


src = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'test_hard'))
labels = os.path.join(os.getcwd(), "output_labels.txt")
input_layer = 'DecodeJpeg/contents:0'
output_layer = 'final_result:0'
num_top_predictions = 1
labels = load_labels(labels)


graph = os.path.join(os.getcwd(), "output_graph_comb.pb")
#graph = os.path.join(os.getcwd(), "output_graph_imgnet.pb")
#graph = os.path.join(os.getcwd(), "output_graph_horea.pb")
load_graph(graph)
run_graph(src, labels, input_layer, output_layer, num_top_predictions)
