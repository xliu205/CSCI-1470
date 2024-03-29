import os
import numpy as np
import tensorflow as tf
from preprocess import *
from transformer_model import Transformer_Seq2Seq
from rnn_model import RNN_Seq2Seq
import sys
import random

from attenvis import AttentionVis
av = AttentionVis()

def train(model, train_french, train_english, eng_padding_index):
	"""
	Runs through one epoch - all training examples.

	:param model: the initialized model to use for forward and backward pass
	:param train_french: french train data (all data for training) of shape (num_sentences, 14)
	:param train_english: english train data (all data for training) of shape (num_sentences, 15)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:return: None
	"""

	# NOTE: For each training step, you should pass in the french sentences to be used by the encoder, 
	# and english sentences to be used by the decoder
	# - The english sentences passed to the decoder have the last token in the window removed:
	#	 [STOP CS147 is the best class. STOP *PAD*] --> [STOP CS147 is the best class. STOP] 
	# 
	# - When computing loss, the decoder labels should have the first word removed:
	#	 [STOP CS147 is the best class. STOP] --> [CS147 is the best class. STOP]
	
	N = train_french.shape[0] // model.batch_size
	for batch in range(N):
		start = batch * model.batch_size
		end = (batch + 1) * model.batch_size
		if (batch + 1) * model.batch_size > train_french.shape[0]:
			end = train_french.shape[0]
		encoder_input = train_french[start: end, :]
		decoder_input = train_english[start: end, 0: ENGLISH_WINDOW_SIZE]
		decoder_label = train_english[start: end, 1: ENGLISH_WINDOW_SIZE + 1]
		mask = (decoder_label != eng_padding_index)

		with tf.GradientTape() as tape:
			prbs = model.call(encoder_input, decoder_input)
			loss = model.loss_function(prbs, decoder_label, mask)

		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

		print('\r', 'train {0:.2f} %'.format((batch + 1) * 100 / N), end='')

	pass

@av.test_func
def test(model, test_french, test_english, eng_padding_index):
	"""
	Runs through one epoch - all testing examples.

	:param model: the initialized model to use for forward and backward pass
	:param test_french: french test data (all data for testing) of shape (num_sentences, 14)
	:param test_english: english test data (all data for testing) of shape (num_sentences, 15)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:returns: a tuple containing at index 0 the perplexity of the test set and at index 1 the per symbol accuracy on test set, 
	e.g. (my_perplexity, my_accuracy)
	"""

	# Note: Follow the same procedure as in train() to construct batches of data!
	
	N = test_french.shape[0] // model.batch_size
	sum_loss = 0
	sum_true = 0
	sum_symbol = 0
	for batch in range(N):
		start = batch * model.batch_size
		end = (batch + 1) * model.batch_size
		if (batch + 1) * model.batch_size > test_french.shape[0]:
			end = test_french.shape[0]
		encoder_input = test_french[start: end, :]
		decoder_input = test_english[start: end, 0: ENGLISH_WINDOW_SIZE]
		decoder_label = test_english[start: end, 1: ENGLISH_WINDOW_SIZE + 1]
		mask = (decoder_label != eng_padding_index)

		prbs = model.call(encoder_input, decoder_input)
		loss = tf.reduce_mean(model.loss_function(prbs, decoder_label, mask)) * (end - start)
		sum_loss += loss
		batch_acc = model.accuracy_function(prbs, decoder_label, mask)
		batch_symbol = np.sum(tf.cast(mask, dtype=tf.float32))
		sum_symbol += batch_symbol
		sum_true += batch_acc * batch_symbol
		

	perplexity = np.exp(sum_loss / test_english.shape[0])
	accuracy = sum_true / sum_symbol

	return perplexity, accuracy

	
	pass

def main():	
	if len(sys.argv) != 2 or sys.argv[1] not in {"RNN","TRANSFORMER"}:
			print("USAGE: python assignment.py <Model Type>")
			print("<Model Type>: [RNN/TRANSFORMER]")
			exit()

	# Change this to "True" to turn on the attention matrix visualization.
	# You should turn this on once you feel your code is working.
	# Note that it is designed to work with transformers that have single attention heads.
	if sys.argv[1] == "TRANSFORMER":
		av.setup_visualization(enable=True)

	
	train_english, test_english, train_french, test_french, english_vocab, french_vocab, eng_padding_index = get_data('../../data/fls.txt','../../data/els.txt','../../data/flt.txt','../../data/elt.txt')
	

	model_args = (FRENCH_WINDOW_SIZE, len(french_vocab), ENGLISH_WINDOW_SIZE, len(english_vocab))
	if sys.argv[1] == "RNN":
		model = RNN_Seq2Seq(*model_args)
	elif sys.argv[1] == "TRANSFORMER":
		model = Transformer_Seq2Seq(*model_args) 
	
	# TODO:
	# Train and Test Model for 1 epoch.
	train(model, train_french, train_english, eng_padding_index)

	perplexity, accuracy = test(model, test_french, test_english, eng_padding_index)
	print('\n\n Perplexity : {0:.2f}'.format(perplexity))
	print('Accuracy : {0:.4f}'.format(accuracy))




	# Visualize a sample attention matrix from the test set
	# Only takes effect if you enabled visualizations above
	av.show_atten_heatmap()
	pass

if __name__ == '__main__':
	main()
