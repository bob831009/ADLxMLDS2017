import json
import numpy as np
import os
import operator
import random
import time
import tensorflow as tf
from keras.preprocessing import sequence

DATA_PATH = './MLDS_hw2_data';

class Video_Caption_Generator():
	def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, n_video_lstm_step, n_caption_lstm_step, bias_init_vector=None):
		self.dim_image = dim_image
		self.n_words = n_words
		self.dim_hidden = dim_hidden
		self.batch_size = batch_size
		self.n_lstm_steps = n_lstm_steps
		self.n_video_lstm_step=n_video_lstm_step
		self.n_caption_lstm_step=n_caption_lstm_step

		with tf.device("/gpu:0"):
			self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')

		self.lstm1 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)
		self.lstm2 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)

		self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
		self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')

		self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')
		if bias_init_vector is not None:
			self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
		else:
			self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

	def build_model(self):
		video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step, self.dim_image])
		video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step])

		caption = tf.placeholder(tf.int32, [self.batch_size, self.n_caption_lstm_step+1])
		caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_lstm_step+1])


		video_flat = tf.reshape(video, [-1, self.dim_image])
		image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b ) # (batch_size*n_lstm_steps, dim_hidden)
		image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden])

		state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
		state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
		padding = tf.zeros([self.batch_size, self.dim_hidden])

		probs = []
		loss = 0.0

		##############################  Encoding Stage ##################################
		for i in range(0, self.n_video_lstm_step):
			if i > 0:
				tf.get_variable_scope().reuse_variables()

			with tf.variable_scope("LSTM1"):
				output1, state1 = self.lstm1(image_emb[:,i,:], state1)

			with tf.variable_scope("LSTM2"):
				output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)

		############################# Decoding Stage ######################################
		for i in range(0, self.n_caption_lstm_step): ## Phase 2 => only generate captions
			with tf.device("/gpu:0"):
				current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])

			tf.get_variable_scope().reuse_variables()

			with tf.variable_scope("LSTM1"):
				output1, state1 = self.lstm1(padding, state1)

			with tf.variable_scope("LSTM2"):
				output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)

			labels = tf.expand_dims(caption[:, i+1], 1)
			indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
			concated = tf.concat([indices, labels], 1)
			onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

			logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logit_words);
			cross_entropy = cross_entropy * caption_mask[:,i]
			probs.append(logit_words)

			current_loss = tf.reduce_sum(cross_entropy)/self.batch_size
			loss = loss + current_loss

		return loss, video, video_mask, caption, caption_mask, probs

	def build_generator(self):
		video = tf.placeholder(tf.float32, [1, self.n_video_lstm_step, self.dim_image])
		video_mask = tf.placeholder(tf.float32, [1, self.n_video_lstm_step])

		video_flat = tf.reshape(video, [-1, self.dim_image])
		image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
		image_emb = tf.reshape(image_emb, [1, self.n_video_lstm_step, self.dim_hidden])

		state1 = tf.zeros([1, self.lstm1.state_size])
		state2 = tf.zeros([1, self.lstm2.state_size])
		padding = tf.zeros([1, self.dim_hidden])

		generated_words = []

		probs = []
		embeds = []

		for i in range(0, self.n_video_lstm_step):
			if i > 0:
				tf.get_variable_scope().reuse_variables()

			with tf.variable_scope("LSTM1"):
				output1, state1 = self.lstm1(image_emb[:, i, :], state1)

			with tf.variable_scope("LSTM2"):
				output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)

		for i in range(0, self.n_caption_lstm_step):
			tf.get_variable_scope().reuse_variables()

			if i == 0:
				with tf.device('/gpu:0'):
					current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([1], dtype=tf.int64))

			with tf.variable_scope("LSTM1"):
				output1, state1 = self.lstm1(padding, state1)

			with tf.variable_scope("LSTM2"):
				output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)

			logit_words = tf.nn.xw_plus_b( output2, self.embed_word_W, self.embed_word_b)
			max_prob_index = tf.argmax(logit_words, 1)[0]
			generated_words.append(max_prob_index)
			probs.append(logit_words)

			with tf.device("/gpu:0"):
				current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
				current_embed = tf.expand_dims(current_embed, 0)

			embeds.append(current_embed)

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
	vocab = [w for w in vocabs_count if vocabs_count[w] >= word_count_threshold]
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
# video_path = '/home/chenxp/data/msvd'

# video_train_feat_path = './rgb_train_features'
# video_test_feat_path = './rgb_test_features'

# video_train_data_path = './data/video_corpus.csv'
# video_test_data_path = './data/video_corpus.csv'

model_path = './models'

#=======================================================================================
# Train Parameters
#=======================================================================================
dim_image = 4096
dim_hidden= 1000

n_video_lstm_step = 80
n_caption_lstm_step = 20
n_frame_step = 80

n_epochs = 1000
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
					bias_init_vector=bias_init_vector);

		tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask, tf_probs = model.build_model()
		sess = tf.InteractiveSession()
		saver = tf.train.Saver()

	train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
	tf.global_variables_initializer().run()

	loss_fd = open('loss.txt', 'w')

	for epoch in range(0, n_epochs):

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

			current_captions = map(lambda x: '<bos> ' + x, current_captions)
			current_captions = map(lambda x: x.replace('.', ''), current_captions)
			current_captions = map(lambda x: x.replace(',', ''), current_captions)
			current_captions = map(lambda x: x.replace('"', ''), current_captions)
			current_captions = map(lambda x: x.replace('\n', ''), current_captions)
			current_captions = map(lambda x: x.replace('?', ''), current_captions)
			current_captions = map(lambda x: x.replace('!', ''), current_captions)
			current_captions = map(lambda x: x.replace('\\', ''), current_captions)
			current_captions = map(lambda x: x.replace('/', ''), current_captions)

			current_captions = list(current_captions);
			for idx, each_cap in enumerate(current_captions):
				word = each_cap.lower().split(' ')
				if len(word) < n_caption_lstm_step:
					current_captions[idx] = current_captions[idx] + ' <eos>'
				else:
					new_word = ''
					for i in range(n_caption_lstm_step-1):
						new_word = new_word + word[i] + ' '
					current_captions[idx] = new_word + '<eos>'

			current_caption_ind = []
			for cap in current_captions:
				current_word_ind = []
				for word in cap.lower().split(' '):
					if word in wordtoix:
						current_word_ind.append(wordtoix[word])
					else:
						current_word_ind.append(wordtoix['<unk>'])
				current_caption_ind.append(current_word_ind)

			current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=n_caption_lstm_step)
			current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix), 1] ) ] ).astype(int)
			current_caption_masks = np.zeros( (current_caption_matrix.shape[0], current_caption_matrix.shape[1]) )
			nonzeros = np.array( list(map(lambda x: (x != 0).sum() + 1, current_caption_matrix )) )

			for ind, row in enumerate(current_caption_masks):
				row[:nonzeros[ind]] = 1

			probs_val = sess.run(tf_probs, feed_dict={
				tf_video:current_feats,
				tf_caption: current_caption_matrix
				})

			_, loss_val = sess.run(
					[train_op, tf_loss],
					feed_dict={
						tf_video: current_feats,
						tf_video_mask : current_video_masks,
						tf_caption: current_caption_matrix,
						tf_caption_mask: current_caption_masks
						})

			print('idx: ', start, " Epoch: ", epoch, " loss: ", loss_val, ' Elapsed time: ', str((time.time() - start_time)))
			loss_fd.write('epoch ' + str(epoch) + ' loss ' + str(loss_val) + '\n')

		if np.mod(epoch, 10) == 0:
			print("Epoch ", epoch, " is done. Saving the model ...")
			saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)

	loss_fd.close();


def test(data_dir, output_file, model_path, special_test=False):
	test_ids = [];
	with open(os.path.join(data_dir, 'testing_id.txt')) as f:
		for line in f:
			test_ids.append(line.strip());

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
		tmp_data = np.load(os.path.join(data_dir, 'testing_data', 'feat', videoId));
		test_data.append(tmp_data);
	test_data = np.array(test_data);

	with open(os.path.join(data_dir, 'training_label.json')) as f:
		train_labels = json.load(f);
	with open(os.path.join(data_dir, 'testing_label.json')) as f:
		test_labels = json.load(f);

	total_labels = train_labels + test_labels;
	wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(total_labels, word_count_threshold=2);

	model = Video_Caption_Generator(
			dim_image=dim_image,
			n_words=len(ixtoword),
			dim_hidden=dim_hidden,
			batch_size=batch_size,
			n_lstm_steps=n_frame_step,
			n_video_lstm_step=n_video_lstm_step,
			n_caption_lstm_step=n_caption_lstm_step,
			bias_init_vector=bias_init_vector)

	video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()

	sess = tf.InteractiveSession()

	saver = tf.train.Saver()
	saver.restore(sess, model_path)

	test_output_txt_fd = open(output_file, 'w')
	for idx, video_feat in enumerate(test_data):
		video_feat = video_feat[None,...];
		if (video_feat.shape[1] == n_frame_step):
			video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
		else:
			continue;

		generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask});
		generated_words = []
		for word_idx in generated_word_index:
			generated_words.append(ixtoword[word_idx]);

		punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
		generated_words = generated_words[:punctuation]

		generated_sentence = ' '.join(generated_words)
		generated_sentence = generated_sentence.replace('<bos> ', '')
		generated_sentence = generated_sentence.replace(' <eos>', '')

		test_output_txt_fd.write('%s,%s\n' % (test_ids[idx], generated_sentence));
