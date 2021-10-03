from __future__ import absolute_import
from matplotlib import pyplot as plt
import numpy as np
from preprocess import get_data

class Model:
    """
    This model class will contain the architecture for
    your single layer Neural Network for classifying mnist with 
    batched learning. Please implement the TODOs for the entire 
    model but do not change the method and constructor arguments. 
    Make sure that your Model class works with multiple batch 
    sizes. Additionally, please exclusively use NumPy and 
    Python built-in functions for your implementation.
    """

    def __init__(self):
        # TODO: Initialize all hyperparametrs
        self.input_size = 784 # Size of image vectors
        self.num_classes = 10 # Number of classes/possible labels
        self.batch_size = 100
        self.learning_rate = 0.4

        # TODO: Initialize weights and biases
        self.W = np.zeros((self.input_size, self.num_classes))
        self.b = np.zeros((1, self.num_classes))

    def call(self, inputs):
        """
        Does the forward pass on an batch of input images.
        :param inputs: normalized (0.0 to 1.0) batch of images,
                       (batch_size x 784) (2D), where batch can be any number.
        :return: probabilities, probabilities for each class per image # (batch_size x 10)
        """
        # TODO: Write the forward pass logic for your model
        # TODO: Calculate, then return, the probability for each class per image using the Softmax equation
        l = np.dot(inputs,self.W + self.b)   
        el = np.exp(l)
        total = np.sum(el, 1)
        total = total.reshape(total.shape[0], 1)  
        probabilities = el / total 
        return probabilities
        pass
    
    
    def loss(self, probabilities, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Loss should be decreasing with every training loop (step). 
        NOTE: This function is not actually used for gradient descent 
        in this assignment, but is a sanity check to make sure model 
        is learning.
        :param probabilities: matrix that contains the probabilities 
        of each class for each image
        :param labels: the true batch labels
        :return: average loss per batch element (float)
        """

        # TODO: calculate average cross entropy loss for a batch

        prob = np.diag(probabilities[:, labels])
        loss = - np.log(prob)
        return np.mean(loss)
        pass
    
    def back_propagation(self, inputs, probabilities, labels):
        """
        Returns the gradients for model's weights and biases 
        after one forward pass and loss calculation. The learning 
        algorithm for updating weights and biases mentioned in 
        class works for one image, but because we are looking at 
        batch_size number of images at each step, you should take the
        average of the gradients across all images in the batch.
        :param inputs: batch inputs (a batch of images)
        :param probabilities: matrix that contains the probabilities of each 
        class for each image
        :param labels: true labels
        :return: gradient for weights,and gradient for biases
        """
        # TODO: calculate the gradients for the weights and the gradients for the bias with respect to average loss
        Y = np.eye(self.batch_size)[labels, 0:self.num_classes]
        Z = (Y - probabilities) / self.batch_size
        gradW =  np.dot(inputs.T, Z)  
        gradB =  np.sum(Z)
             
        return gradW, gradB
        pass
    
    def accuracy(self, probabilities, labels):
        """
        Calculates the model's accuracy by comparing the number 
        of correct predictions with the correct answers.
        :param probabilities: result of running model.call() on test inputs
        :param labels: test set labels
        :return: Float (0,1) that contains batch accuracy
        """
        # TODO: calculate the batch accuracy
        prediction = np.argmax(probabilities, axis=1) 
        return sum(prediction == labels) / labels.shape[0]
        pass

    def gradient_descent(self, gradW, gradB):
        '''
        Given the gradients for weights and biases, does gradient 
        descent on the Model's parameters.
        :param gradW: gradient for weights
        :param gradB: gradient for biases
        :return: None
        '''
        # TODO: change the weights and biases of the model to descent the gradient
        self.W += gradW * self.learning_rate
        self.b += gradB * self.learning_rate
        pass
    
def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels.
    :param model: the initialized model to use for the forward 
    pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training)
    :param train_inputs: train labels (all labels to use for training)
    :return: None
    '''
    # TODO: Iterate over the training inputs and labels, in model.batch_size increments
    # TODO: For every batch, compute then descend the gradients for the model's weights
    # Optional TODO: Call visualize_loss and observe the loss per batch as the model trains.
    for i in range(0, int(train_inputs.shape[0]/model.batch_size)):
        
        inputs = train_inputs[model.batch_size*i : model.batch_size*(i+1), :]
        labels = train_labels[model.batch_size*i : model.batch_size*(i+1)]
        probabilities = model.call(inputs)
        gradW, gradB = model.back_propagation(inputs, probabilities, labels)

        model.gradient_descent(gradW, gradB)
    pass

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. For this assignment, 
    the inputs should be the entire test set, but in the future we will
    ask you to batch it instead.
    :param test_inputs: mnist test data (all images to be tested)
    :param test_labels: mnist test labels (all corresponding labels)
    :return: accuracy - Float (0,1)
    """
    # TODO: Iterate over the testing inputs and labels
    # TODO: Return accuracy across testing set
    probabilities = model.call(test_inputs) 
    return model.accuracy(probabilities, test_labels)
    pass

def visualize_loss(losses):
    """
    Uses Matplotlib to visualize loss per batch. Call this in train().
    When you observe the plot that's displayed, think about:
    1. What does the plot demonstrate or show?
    2. How long does your model need to train to reach roughly its best accuracy so far, 
    and how do you know that?
    Optionally, add your answers to README!
    param losses: an array of loss value from each batch of train

    NOTE: DO NOT EDIT
    
    :return: doesn't return anything, a plot should pop-up
    """
    x = np.arange(1, len(losses)+1)
    plt.xlabel('i\'th Batch')
    plt.ylabel('Loss Value')
    plt.title('Loss per Batch')
    plt.plot(x, losses)
    plt.show()

def visualize_results(image_inputs, probabilities, image_labels):
    """
    Uses Matplotlib to visualize the results of our model.
    :param image_inputs: image data from get_data()
    :param probabilities: the output of model.call()
    :param image_labels: the labels from get_data()

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    images = np.reshape(image_inputs, (-1, 28, 28))
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = images.shape[0]

    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
        ax.imshow(images[ind], cmap="Greys")
        ax.set(title="PL: {}\nAL: {}".format(predicted_labels[ind], image_labels[ind]))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    plt.show()

def main():
    '''
    Read in mnist data, initialize your model, and train and test your model 
    for one epoch. The number of training steps should be your the number of 
    batches you run through in a single epoch. You should receive a final accuracy on the testing examples of > 80%.
    :return: None
    '''

    # TODO: load mnist train and test examples into train_inputs, train_labels, test_inputs, test_labels
    train_inputs, train_labels = \
        get_data('./mnist_data/train-images-idx3-ubyte.gz', './mnist_data/train-labels-idx1-ubyte.gz', 60000)
        #trainset = 60000
    test_inputs, test_labels = \
        get_data('./mnist_data/t10k-images-idx3-ubyte.gz', './mnist_data/t10k-labels-idx1-ubyte.gz', 10000)
        #testset = 10000

    # TODO: Create Model
    model = Model()

    # TODO: Train model by calling train() ONCE on all data
    train(model, train_inputs, train_labels)

    # TODO: Test the accuracy by calling test() after running train()
    print('model accuracy =', test(model, test_inputs, test_labels))

    # TODO: Visualize the data by using visualize_results()
    # take the first 10 images for visualization
    input = test_inputs[0:10, :]
    label = test_labels[0:10]
    visualize_results(input, model.call(input), label)

    pass
    
if __name__ == '__main__':
    main()
