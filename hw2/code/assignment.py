from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d

import os
import tensorflow as tf
import numpy as np
import random
import math

class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.batch_size = 100
        self.num_classes = 2
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main

        # TODO: Initialize all hyperparameters
        self.learnig_rate = 1e-3
        self.optimizer = tf.keras.optimizers.Adam(self.learnig_rate)

        self.epoch = 10

        self.filter_num1 = 2
        self.stride_size1 = 1
        self.pool1_ksize = 2
        self.pool1_stride = 2

        self.filter_num2 = 16
        self.stride_size2 = 1
        self.pool2_ksize = 2
        self.pool2_stride = 2

        self.filter_num3 = 64
        self.stride_size3 = 1
        self.pool3_ksize = 2
        self.pool3_stride = 2

        self.flatten_width = 1024
        self.dense1_output_width = 4
        self.dense2_output_width = 2
       

        # TODO: Initialize all trainable parameters
        self.filter1 = tf.Variable(tf.random.truncated_normal([3, 3, 3, self.filter_num1], stddev=0.1))
        self.stride1 = [1, self.stride_size1, self.stride_size1, 1]
        self.filter2 = tf.Variable(tf.random.truncated_normal([3, 3, self.filter_num1, self.filter_num2], stddev=0.1))
        self.stride2 = [1, self.stride_size2, self.stride_size2, 1]
        self.filter3 = tf.Variable(tf.random.truncated_normal([3, 3, self.filter_num2, self.filter_num3], stddev=0.1))
        self.stride3 = [1, self.stride_size3, self.stride_size3, 1]
        

        self.w1 = tf.Variable(tf.random.normal([self.flatten_width, self.dense1_output_width], stddev=.1, dtype=tf.float32))
        self.w2 = tf.Variable(tf.random.normal([self.dense1_output_width, self.dense2_output_width], stddev=.1, dtype=tf.float32))
        self.w3 = tf.Variable(tf.random.normal([self.dense2_output_width, self.num_classes], stddev=.1, dtype=tf.float32))


        self.b1 = tf.Variable(tf.random.normal([1, self.dense1_output_width], stddev=.1, dtype=tf.float32))
        self.b2 = tf.Variable(tf.random.normal([1, self.dense2_output_width], stddev=.1, dtype=tf.float32))
        self.b3 = tf.Variable(tf.random.normal([1, self.num_classes], stddev=.1, dtype=tf.float32))
		

    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)
 
        conv1 = tf.nn.conv2d(inputs, self.filter1, self.stride1, 'SAME')
        mean1, variance1 = tf.nn.moments(conv1, axes=[0, 1, 2])
        norm1 = tf.nn.batch_normalization(conv1, mean1, variance1, offset=None, scale=None, variance_epsilon=1e-3)
        relu1 = tf.nn.relu(norm1)
        pool1 = tf.nn.max_pool(relu1, self.pool1_ksize, self.pool1_stride, 'SAME')

        conv2 = tf.nn.conv2d(pool1, self.filter2, self.stride2, 'SAME')
        mean2, variance2 = tf.nn.moments(conv2, axes=[0, 1, 2])
        norm2 = tf.nn.batch_normalization(conv2, mean2, variance2, offset=None, scale=None, variance_epsilon=1e-3)
        relu2 = tf.nn.relu(norm2)
        pool2 = tf.nn.max_pool(relu2, self.pool2_ksize, self.pool2_stride, 'SAME')

    
        conv3 = tf.nn.conv2d(pool2, self.filter3, self.stride3, 'SAME')
        mean3, variance3 = tf.nn.moments(conv3, axes=[0, 1, 2])
        norm3 = tf.nn.batch_normalization(conv3, mean3, variance3, offset=None, scale=None, variance_epsilon=1e-3)
        relu3 = tf.nn.relu(norm3)
        pool3 = tf.nn.max_pool(relu3, self.pool3_ksize, self.pool3_stride, 'SAME')
  
        dense_input = tf.reshape(pool3, [-1, self.flatten_width])

        dense_l1 = tf.nn.relu(tf.matmul(dense_input, self.w1) + self.b1)
        dense_l1 = tf.nn.dropout(dense_l1, rate=0.3)

        dense_l2 = tf.nn.relu(tf.matmul(dense_l1, self.w2) + self.b2)
        dense_l2 = tf.nn.dropout(dense_l2, rate=0.3)

        dense_l3 = tf.nn.relu(tf.matmul(dense_l2, self.w3) + self.b3)
        

        logits = dense_l3
        return logits

        pass

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        return tf.nn.softmax_cross_entropy_with_logits(labels, logits)
        pass

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT
        
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
    and labels - ensure that they are shuffled in the same order using tf.gather or zipping.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''
    # shuffle
    indices = tf.range(0, train_inputs.shape[0])
    indices = tf.random.shuffle(indices)
    train_inputs = tf.gather(train_inputs, indices)
    train_labels = tf.gather(train_labels, indices)

    for batch in range(int(train_inputs.shape[0]/model.batch_size)):
        start = batch * model.batch_size
        end = (batch + 1) * model.batch_size
        if (batch + 1) * model.batch_size > train_inputs.shape[0]:  
            end = train_inputs.shape[0]
        inputs = tf.image.random_flip_left_right(train_inputs[start: end])  
        labels = train_labels[start: end]

        with tf.GradientTape() as tape:
            logits = model.call(inputs)
            loss = model.loss(logits, labels)

            if batch % 10 == 0:  # print training accuracy every 10 batches
                train_acc = model.accuracy(logits, labels)
                print("Accuracy on training set after {} images: {}".format(model.batch_size * batch, train_acc))

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    pass

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    test_logits = model.call(test_inputs, is_testing=True)
    test_accuracy = model.accuracy(test_logits, test_labels)
    pass


def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"

    NOTE: DO NOT EDIT

    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label): 
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle("{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
        
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images): 
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0): 
            correct.append(i)
        else: 
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()


def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs. 
    
    CS1470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=70%.
    
    CS2470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=75%.
    
    :return: None
    '''

    first_class = 3
    second_class = 5
    train_inputs, train_labels = get_data('./data/train', first_class, second_class)
    test_inputs, test_labels = get_data('./data/test', first_class, second_class)

    print("train set size: ", train_inputs.shape, train_labels.shape)
    print("test set size: ",test_inputs.shape, test_labels.shape)

    model = Model()

    for epoch in range(0, model.epoch):
        print("\n       \     epoch {}     \      ".format(epoch))
        train(model, train_inputs, train_labels)
    print("\n   \    end     \    \n")

    test_accuracy = test(model, test_inputs, test_labels)
    print("Accuracy on test set is: {}".format(test_accuracy))

    sample_inputs = test_inputs[0:10]
    sample_labels = test_labels[0:10]
    sample_logits = model.call(sample_inputs, sample_labels)
    visualize_results(sample_inputs, sample_logits, sample_labels, 'cat', 'dog')
    return


if __name__ == '__main__':
    main()
