import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import get_data


class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the next words in a sequence.

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # TODO: initialize vocab_size, embedding_size

        self.vocab_size = vocab_size
        self.window_size = 20 # DO NOT CHANGE!
        self.embedding_size = 300 #TODO
        self.batch_size = 300 #TODO 
        self.rnn_size = 256  
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.03)

        # TODO: initialize embeddings and forward pass weights (weights, biases)
        # Note: You can now use tf.keras.layers!
        # - use tf.keras.layers.Dense for feed forward layers: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        # - and use tf.keras.layers.GRU or tf.keras.layers.LSTM for your RNN 
        self.E = tf.Variable(tf.random.truncated_normal(shape=[self.vocab_size, self.embedding_size], mean=0, stddev=0.1))
        self.lstm = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(self.vocab_size, activation='softmax')

    def call(self, inputs, initial_state):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.

        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, a final_state (Note 1: If you use an LSTM, the final_state will be the last two RNN outputs, 
        Note 2: We only need to use the initial state during generation)
        using LSTM and only the probabilites as a tensor and a final_state as a tensor when using GRU 
        """
        
        #TODO: Fill in 
        embedding = tf.nn.embedding_lookup(self.E, inputs) 
        output, state1, state2 = self.lstm(embedding, initial_state=initial_state)
        dense = self.dense(output)

        return dense, (state1, state2)
        

    def loss(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        
        NOTE: You have to use np.reduce_mean and not np.reduce_sum when calculating your loss

        :param logits: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """

        #TODO: Fill in
        #We recommend using tf.keras.losses.sparse_categorical_crossentropy
        #https://www.tensorflow.org/api_docs/python/tf/keras/losses/sparse_categorical_crossentropy

        return tf.keras.losses.sparse_categorical_crossentropy(labels, probs)


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    #TODO: Fill in
   
    num_inputs = int(train_inputs.shape[0]/model.window_size)
   
    train_inputs = np.reshape(train_inputs[:num_inputs*model.window_size],(-1,model.window_size))
    train_labels = np.reshape(train_labels[:num_inputs*model.window_size],(-1,model.window_size))

  
    for i in range(int(train_inputs.shape[0]/model.batch_size)):
        start = i * model.batch_size
        end = (i + 1) * model.batch_size
        if (i + 1) * model.batch_size > train_inputs.shape[0]:
            end = train_inputs.shape[0]
        inputs = train_inputs[start: end]
        labels = train_labels[start: end]

        with tf.GradientTape() as tape:
            probs, _ = model.call(inputs, None)
            loss = model.loss(probs, labels)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
   

    pass


def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set
    """
    
    #TODO: Fill in
    #NOTE: Ensure a correct perplexity formula (different from raw loss)
    
    num_inputs = int(test_inputs.shape[0]/model.window_size)
   
    test_inputs = np.reshape(test_inputs[:num_inputs*model.window_size],(-1,model.window_size))
    test_labels = np.reshape(test_labels[:num_inputs*model.window_size],(-1,model.window_size))
    loss = 0.0
    batch_num = 0.0
    for i in range(int(test_inputs.shape[0]/model.batch_size)):
        start = i 
        end = min(test_inputs.shape[0],i+model.batch_size) 
        
        test_input0 = test_inputs[start:end]
        test_label0 = test_labels[start:end]
        probs = model.call(test_input0, None)
        loss+= model.loss(probs, test_label0)
        batch_num += 1
    avg_loss = loss/batch_num
    return np.exp(avg_loss)

    pass  


def generate_sentence(word1, length, vocab, model, sample_n=10):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    """

    #NOTE: Feel free to play around with different sample_n values

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    for i in range(length):
        logits, previous_state = model.call(next_input, previous_state)
        logits = np.array(logits[0,0,:])
        top_n = np.argsort(logits)[-sample_n:]
        n_logits = np.exp(logits[top_n])/np.exp(logits[top_n]).sum()
        out_index = np.random.choice(top_n,p=n_logits)

        text.append(reverse_vocab[out_index])
        next_input = [[out_index]]

    print(" ".join(text))


def main():
    # TO-DO: Pre-process and vectorize the data
    train_token, test_token, vocab_dict = get_data('data/train.txt', 'data/test.txt')
    train_token=np.array(train_token)
    test_token=np.array(test_token)
   
    # HINT: Please note that you are predicting the next word at each timestep, so you want to remove the last element
    # from train_x and test_x. You also need to drop the first element from train_y and test_y.
    # If you don't do this, you will see impossibly small perplexities.

    
    # TO-DO:  Separate your train and test data into inputs and labels
    train_inputs=train_token[:-1]
    train_labels=train_token[1:]
    test_inputs=test_token[:-1]
    test_labels=test_token[1:]
    # TODO: initialize model
    model = Model(len(vocab_dict))
    
    # TODO: Set-up the training step
    train(model, train_inputs, train_labels)

    # TODO: Set up the testing steps

    # Print out perplexity 
    Perplexity = test(model, test_inputs, test_labels)
    print('Perplexity = {}'.format(Perplexity))

    

    # BONUS: Try printing out various sentences with different start words and sample_n parameters 
    
    pass

if __name__ == '__main__':
    main()
