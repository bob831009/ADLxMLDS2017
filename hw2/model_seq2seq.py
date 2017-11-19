#-*- coding: utf-8 -*-
import json
import numpy as np
import os
import operator
import random
import time
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops import variable_scope
from keras.preprocessing import sequence

DATA_PATH = './MLDS_hw2_data';

class Video_Caption_Generator():
	def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, n_video_lstm_step, n_caption_lstm_step, max_gradient_norm, learning_rate, bias_init_vector=None):
		self.dim_image = dim_image
		self.n_words = n_words
		self.dim_hidden = dim_hidden
		self.batch_size = batch_size
		self.n_lstm_steps = n_lstm_steps
		self.n_video_lstm_step = n_video_lstm_step
		self.n_caption_lstm_step = n_caption_lstm_step

		self.max_gradient_norm = max_gradient_norm
		self.learning_rate = learning_rate
		self.num_layers = 2;

	def build_model(self):
		video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step, self.dim_image])
		video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step])

		caption = tf.placeholder(tf.int32, [self.batch_size, self.n_caption_lstm_step+1])
		caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_lstm_step+1])

		caption_target = tf.placeholder(tf.int32, [self.batch_size, self.n_caption_lstm_step+1]);


		# Build RNN cell
		# encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.dim_hidden)
		forward_encoder_cell = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(self.dim_hidden) for _ in range(self.num_layers)])
		backward_encoder_cell = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(self.dim_hidden) for _ in range(self.num_layers)])

		# Run Dynamic RNN
		#   encoder_outpus: [batch_size, max_time, num_units]
		#   encoder_state: [batch_size, num_units]
		# encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, video, dtype=tf.float32, time_major=False)
		(encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_state, encoder_bw_state) = tf.nn.bidirectional_dynamic_rnn(forward_encoder_cell, backward_encoder_cell, video, dtype=tf.float32, time_major=False)
		encoder_outputs = tf.add(encoder_fw_outputs, encoder_bw_outputs);


		encoder_state = []
		for i in range(self.num_layers):
			if isinstance(encoder_fw_state[i],tf.contrib.rnn.LSTMStateTuple):
				encoder_state_c = tf.add(encoder_fw_state[i].c, encoder_bw_state[i].c)
				encoder_state_h = tf.add(encoder_fw_state[i].h,encoder_bw_state[i].h)
				tmp_encoder_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
			elif isinstance(encoder_fw_state[i], tf.Tensor):
				tmp_encoder_state = tf.add(encoder_fw_state[i], encoder_bw_state[i])
			encoder_state.append(tmp_encoder_state)
		encoder_state = tuple(encoder_state)

		# attention
		attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.dim_hidden, encoder_outputs)

		# Embedding
		with tf.variable_scope('embedding'):
			embedding_decoder = tf.Variable(tf.truncated_normal(shape=[self.n_words, self.dim_hidden], stddev=0.1), name='embedding_decoder')
		# embedding_decoder = variable_scope.get_variable("embedding_decoder", [self.n_words, self.dim_hidden]);
		
		with tf.device("/cpu:0"):			
			# Look up embedding:
			#   decoder_inputs: [batch_size, max_time]
			#   decoder_emb_inp: [batch_size, max_time, embedding_size]
			decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, caption)

		# Build RNN cell
		# decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.dim_hidden)
		decoder_cell = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(self.dim_hidden) for _ in range(self.num_layers)])

		decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size=self.dim_hidden)

		# Helper
		decoder_seq_length = [self.n_caption_lstm_step+1] * self.batch_size;
		# helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(decoder_emb_inp, decoder_seq_length, embedding_decoder, 0.3, time_major=False)
		helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, decoder_seq_length, time_major=False)

		# Decoder
		projection_layer = Dense(self.n_words, use_bias=False)
		decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, 
			decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=encoder_state),
			output_layer=projection_layer)
		
		# Dynamic decoding
		outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
		logits = outputs.rnn_output
		translate = outputs.sample_id

		crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=caption_target, logits=logits)
		train_loss = (tf.reduce_sum(crossent * caption_mask) / self.batch_size)

		# Calculate and clip gradients
		params = tf.trainable_variables()
		gradients = tf.gradients(train_loss, params)
		clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

		# Optimization
		optimizer = tf.train.AdamOptimizer(self.learning_rate)
		update_step = optimizer.apply_gradients(zip(clipped_gradients, params))


		return video, video_mask, caption, caption_mask, caption_target, train_loss, update_step, translate

		
	def build_generator(self):
		video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step, self.dim_image])
		video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step])

		generated_words = []
		probs = []
		embeds = []


		# Build RNN cell
		# encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.dim_hidden)
		forward_encoder_cell = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(self.dim_hidden) for _ in range(self.num_layers)])
		backward_encoder_cell = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(self.dim_hidden) for _ in range(self.num_layers)])

		# Run Dynamic RNN
		#   encoder_outpus: [batch_size, max_time, batch_size, num_units]
		#   encoder_state: [batch_size, num_units]
		# encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, video, dtype=tf.float32, time_major=False)
		(encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_state, encoder_bw_state) = tf.nn.bidirectional_dynamic_rnn(forward_encoder_cell, backward_encoder_cell, video, dtype=tf.float32, time_major=False)
		encoder_outputs = tf.add(encoder_fw_outputs, encoder_bw_outputs);


		encoder_state = []
		for i in range(self.num_layers):
			if isinstance(encoder_fw_state[i],tf.contrib.rnn.LSTMStateTuple):
				encoder_state_c = tf.add(encoder_fw_state[i].c, encoder_bw_state[i].c)
				encoder_state_h = tf.add(encoder_fw_state[i].h,encoder_bw_state[i].h)
				tmp_encoder_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
			elif isinstance(encoder_fw_state[i], tf.Tensor):
				tmp_encoder_state = tf.add(encoder_fw_state[i], encoder_bw_state[i])
			encoder_state.append(tmp_encoder_state)
		encoder_state = tuple(encoder_state)

		# attention
		attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.dim_hidden, encoder_outputs)

		# Embedding
		with tf.variable_scope('embedding'):
			embedding_decoder = tf.Variable(tf.truncated_normal(shape=[self.n_words, self.dim_hidden], stddev=0.1), name='embedding_decoder')
		# embedding_decoder = variable_scope.get_variable("embedding_decoder", [self.n_words, self.dim_hidden]);

		# Build RNN cell
		# decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.dim_hidden)
		decoder_cell = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(self.dim_hidden) for _ in range(self.num_layers)])

		decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size=self.dim_hidden)

		# Helper
		helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_decoder, tf.fill([self.batch_size], 1), 2)
		
		# Decoder
		projection_layer = Dense(self.n_words, use_bias=False)
		decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, 
			decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=encoder_state),
			output_layer=projection_layer)
		
		# Dynamic decoding
		outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=self.n_caption_lstm_step);
		translations = outputs.sample_id;

		generated_words = translations;
		
		return video, video_mask, generated_words, probs, embeds

def remove_redundent_notation(captions):
	captions = map(lambda x: x.replace('.', ''), captions)
	captions = map(lambda x: x.replace(',', ''), captions)
	captions = map(lambda x: x.replace('"', ''), captions)
	captions = map(lambda x: x.replace('\n', ''), captions)
	captions = map(lambda x: x.replace('?', ''), captions)
	captions = map(lambda x: x.replace('!', ''), captions)
	captions = map(lambda x: x.replace('\\', ''), captions)
	captions = map(lambda x: x.replace('/', ''), captions)
	return captions;

def preProBuildWordVocab(labels, word_count_threshold=5):
	# borrowed this function from NeuralTalk
	print('preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold))
	vocabs_count = {}
	nsents = 0

	for label in labels:
		captions = remove_redundent_notation(label['caption']);
		for cap in captions:
			nsents += 1;
			for word in cap.lower().split(' '):
				if(word not in vocabs_count): vocabs_count[word] = 0;
				vocabs_count[word] += 1;

	# ensure word index is same all the times
	vocab = [];
	for w in sorted(vocabs_count.keys()):
		if(vocabs_count[w] >= word_count_threshold):
			vocab.append(w);
	print('filtered words from %d to %d' % (len(vocabs_count), len(vocab)));

	ixtoword = {}
	ixtoword[0] = '<pad>'
	ixtoword[1] = '<bos>'
	ixtoword[2] = '<eos>'
	ixtoword[3] = '<unk>'

	wordtoix = {}
	wordtoix['<pad>'] = 0
	wordtoix['<bos>'] = 1
	wordtoix['<eos>'] = 2
	wordtoix['<unk>'] = 3

	for idx, w in enumerate(vocab):
		wordtoix[w] = idx+4
		ixtoword[idx+4] = w

	vocabs_count['<pad>'] = nsents
	vocabs_count['<bos>'] = nsents
	vocabs_count['<eos>'] = nsents
	vocabs_count['<unk>'] = nsents

	bias_init_vector = np.array([1.0 * vocabs_count[ ixtoword[i] ] for i in ixtoword])
	bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
	bias_init_vector = np.log(bias_init_vector)
	bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range

	return wordtoix, ixtoword, bias_init_vector

#=====================================================================================
# Global Parameters
#=====================================================================================
model_path = './models'

#=======================================================================================
# Train Parameters
#=======================================================================================
dim_image = 4096
dim_hidden= 256

n_video_lstm_step = 80
n_caption_lstm_step = 20
n_frame_step = 80

n_epochs = 100
batch_size = 50
learning_rate = 0.0001

def train():
	with open(os.path.join(DATA_PATH, 'training_label.json')) as f:
		train_labels = json.load(f);
	with open(os.path.join(DATA_PATH, 'testing_label.json')) as f:
		test_labels = json.load(f);

	total_labels = train_labels + test_labels;
	wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(total_labels, word_count_threshold=2);

	train_data = [];
	for data in train_labels:
		videoId = '%s.npy' % (data['id']);
		tmp_data = np.load(os.path.join(DATA_PATH, 'training_data', 'feat', videoId));
		train_data.append(tmp_data);
	train_data = np.array(train_data);

	with tf.variable_scope(tf.get_variable_scope()):
		model = Video_Caption_Generator(
					dim_image=dim_image,
					n_words=len(wordtoix),
					dim_hidden=dim_hidden,
					batch_size=batch_size,
					n_lstm_steps=n_frame_step,
					n_video_lstm_step=n_video_lstm_step,
					n_caption_lstm_step=n_caption_lstm_step,
					bias_init_vector=bias_init_vector,
					max_gradient_norm=5,
					learning_rate=learning_rate);

		tf_video, tf_video_mask, tf_caption, tf_caption_mask, tf_caption_target, tf_train_loss, tf_update_step, tf_translate = model.build_model()
		sess = tf.InteractiveSession()
		saver = tf.train.Saver(max_to_keep=10)

	# train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
	tf.global_variables_initializer().run()

	loss_fd = open('loss.txt', 'w')

	for epoch in range(0, n_epochs + 1):

		for start, end in zip(
				range(0, len(train_data), batch_size),
				range(batch_size, len(train_data), batch_size)):

			start_time = time.time()

			current_feats = train_data[start:end]
			current_video_masks = np.zeros((batch_size, n_video_lstm_step))
			current_captions = [];

			for ind in range(len(current_feats)):
				current_video_masks[ind][:len(current_feats[ind])] = 1
				current_captions.append(random.choice(train_labels[start + ind]['caption']));

			current_captions = remove_redundent_notation(current_captions);
			# print(list(current_captions))
			current_captions = list(current_captions);

			current_captions_src = [];
			current_captions_target = [];

			for idx, each_cap in enumerate(current_captions):
				# handle source caption
				current_captions_src.append('<bos> ' + each_cap);

				# handle target caption
				word = each_cap.lower().split(' ')
				if len(word) < n_caption_lstm_step:
					current_captions_target.append(each_cap + ' <eos>');
				else:
					new_word = ''
					for i in range(n_caption_lstm_step-1):
						new_word = new_word + word[i] + ' '
					current_captions_target.append(new_word + '<eos>');

			current_caption_ind_src = []
			for cap in current_captions_src:
				current_word_ind = []
				for word in cap.lower().split(' '):
					if word in wordtoix:
						current_word_ind.append(wordtoix[word])
					else:
						current_word_ind.append(wordtoix['<unk>'])
				current_caption_ind_src.append(current_word_ind)

			current_caption_ind_target = []
			for cap in current_captions_target:
				current_word_ind = []
				for word in cap.lower().split(' '):
					if word in wordtoix:
						current_word_ind.append(wordtoix[word])
					else:
						current_word_ind.append(wordtoix['<unk>'])
				current_caption_ind_target.append(current_word_ind)

			current_caption_matrix_src = sequence.pad_sequences(current_caption_ind_src, padding='post', maxlen=n_caption_lstm_step)
			current_caption_matrix_src = np.hstack( [current_caption_matrix_src, np.zeros( [len(current_caption_matrix_src), 1] ) ] ).astype(int)
			
			current_caption_matrix_target = sequence.pad_sequences(current_caption_ind_target, padding='post', maxlen=n_caption_lstm_step)
			current_caption_matrix_target = np.hstack( [current_caption_matrix_target, np.zeros( [len(current_caption_matrix_target), 1] ) ] ).astype(int)


			current_caption_masks = np.zeros( (current_caption_matrix_src.shape[0], current_caption_matrix_src.shape[1]) )
			nonzeros = np.array( list(map(lambda x: (x != 0).sum() + 1, current_caption_matrix_src )) )

			for ind, row in enumerate(current_caption_masks):
				row[:nonzeros[ind]] = 1

			update_step = sess.run(tf_update_step, feed_dict={
				tf_video: current_feats,
				tf_video_mask : current_video_masks,
				tf_caption: current_caption_matrix_src,
				tf_caption_mask: current_caption_masks,
				tf_caption_target: current_caption_matrix_target
				})

			loss_val, translate = sess.run([tf_train_loss, tf_translate], feed_dict={
				tf_video: current_feats,
				tf_video_mask : current_video_masks,
				tf_caption: current_caption_matrix_src,
				tf_caption_mask: current_caption_masks,
				tf_caption_target: current_caption_matrix_target
				})

			print('idx: ', start, " Epoch: ", epoch, " loss: ", loss_val, ' Elapsed time: ', str((time.time() - start_time)))
			loss_fd.write('epoch ' + str(epoch) + ' loss ' + str(loss_val) + '\n')

		if np.mod(epoch, 10) == 0:
			print("Epoch ", epoch, " is done. Saving the model ...")
			saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)

	loss_fd.close();


def test(data_dir, output_file, peer_output_file, model_path, special_test=False):
	test_id_file = 'testing_id.txt';
	test_data_dir = 'testing_data';

	peer_id_file = 'peer_review_id.txt';
	peer_data_dir = 'peer_review';

	test_ids = [];
	with open(os.path.join(data_dir, test_id_file)) as f:
		for line in f:
			test_ids.append(line.strip());

	peer_ids = [];
	with open(os.path.join(data_dir, peer_id_file)) as f:
		for line in f:
			peer_ids.append(line.strip());

	if(special_test): 
		test_ids = [
		'klteYv1Uv9A_27_33.avi',
		'5YJaS2Eswg0_22_26.avi',
		'UbmZAe5u5FI_132_141.avi',
		'JntMAcTlOF0_50_70.avi',
		'tJHUH9tpqPg_113_118.avi'];

	test_data = [];
	for test_id in test_ids:
		videoId = '%s.npy' % (test_id);
		tmp_data = np.load(os.path.join(data_dir, test_data_dir, 'feat', videoId));
		test_data.append(tmp_data);
	# test_data = np.array(test_data);

	peer_data = [];
	for peer_id in peer_ids:
		videoId = '%s.npy' % (peer_id);
		tmp_data = np.load(os.path.join(data_dir, peer_data_dir, 'feat', videoId));
		peer_data.append(tmp_data);
	# peer_data = np.array(peer_data);

	total_data_data = np.array(test_data + peer_data);

	with open(os.path.join(data_dir, 'training_label.json')) as f:
		train_labels = json.load(f);
	with open(os.path.join(data_dir, 'testing_label.json')) as f:
		test_labels = json.load(f);

	total_labels = train_labels + test_labels;
	wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(total_labels, word_count_threshold=2);


	with tf.variable_scope(tf.get_variable_scope()):
		model = Video_Caption_Generator(
				dim_image=dim_image,
				n_words=len(ixtoword),
				dim_hidden=dim_hidden,
				batch_size=len(total_data_data),
				n_lstm_steps=n_frame_step,
				n_video_lstm_step=n_video_lstm_step,
				n_caption_lstm_step=n_caption_lstm_step,
				bias_init_vector=bias_init_vector,
				max_gradient_norm=5,
				learning_rate=learning_rate)

		video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()

		sess = tf.InteractiveSession()

		saver = tf.train.Saver()
		saver.restore(sess, model_path)

	test_output_txt_fd = open(output_file, 'w');
	peer_output_txt_fd = open(peer_output_file, 'w');
	generated_word_indexs = sess.run(caption_tf, feed_dict={video_tf:total_data_data});
	for idx, generated_word_index in enumerate(generated_word_indexs):
		generated_words = []
		for word_idx in generated_word_index:
			generated_words.append(ixtoword[word_idx]);

		punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
		generated_words = generated_words[:punctuation]
		generated_sentence = ' '.join(generated_words)
		generated_sentence = generated_sentence.replace('<bos> ', '')
		generated_sentence = generated_sentence.replace(' <eos>', '')
		print(generated_sentence)
		if(idx < len(test_data)):
			test_output_txt_fd.write('%s,%s\n' % (test_ids[idx], generated_sentence));
		else:
			peer_output_txt_fd.write('%s,%s\n' % (peer_ids[idx - len(test_data)], generated_sentence));
