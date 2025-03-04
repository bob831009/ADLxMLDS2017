import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
import time
import os
# from model_resnet import Generator, Discriminator
from model_acgan_resnet import Generator, Discriminator
# import progressbar as pb
import data_utils

def get_continue_sum(a):
	output = 0
	for i in range(a+1):
		output += i
	return output

class ACGAN(object):
	def __init__(self, data, vocab_processor, FLAGS, test=False):
		config = tf.ConfigProto(allow_soft_placement = True)
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config = config)
		self.data = data
		self.vocab_processor = vocab_processor
		self.vocab_size = len(vocab_processor._reverse_mapping)
		self.FLAGS = FLAGS
		self.img_row = self.data.img_feat.shape[1]
		self.img_col = self.data.img_feat.shape[2]
		self.alpha = 10.
		self.d_epoch = 1
		self.la = 5
		if(not test): self.gen_path()

	def gen_path(self):
		# Output directory for models and summaries
		timestamp = str(time.strftime('%b-%d-%Y-%H-%M-%S'))
		self.out_dir = os.path.abspath(os.path.join(os.path.curdir, "models", timestamp))
		print ("Writing to {}\n".format(self.out_dir))
	    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
		self.checkpoint_dir = os.path.abspath(os.path.join(self.out_dir, "checkpoints"))
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "model")
		if not os.path.exists(self.checkpoint_dir):
			os.makedirs(self.checkpoint_dir)

	def build_model(self):

		self.g_net = Generator( 
						max_seq_length=self.data.tags_idx.shape[1], 
						vocab_size=self.vocab_size, 
						embedding_size=self.FLAGS.embedding_dim, 
						hidden_size=self.FLAGS.hidden,
						img_row=self.img_row,
						img_col=self.img_col)
		self.d_net = Discriminator( 
						max_seq_length=self.data.tags_idx.shape[1], 
						vocab_size=self.vocab_size, 
						embedding_size=self.FLAGS.embedding_dim, 
						hidden_size=self.FLAGS.hidden,
						img_row=self.img_row,
						img_col=self.img_col)

		self.seq = tf.placeholder(tf.float32, [None, len(self.data.eyes_idx)+len(self.data.hair_idx)], name="seq")
		self.img = tf.placeholder(tf.float32, [None, self.img_row, self.img_col, 3], name="img")
		self.z = tf.placeholder(tf.float32, [None, self.FLAGS.z_dim])

		self.w_seq = tf.placeholder(tf.float32, [None, len(self.data.eyes_idx)+len(self.data.hair_idx)], name="w_seq")
		self.w_img = tf.placeholder(tf.float32, [None, self.img_row, self.img_col, 3], name="w_img")

		# print(self.seq.shape)
		# self.rand_seq = tf.placeholder(tf.float32, [None, len(self.data.eyes_idx)+len(self.data.hair_idx)], name="rand_seq")

		# r_img, r_seq = self.img, self.seq

		# self.f_img = self.g_net(r_seq, self.z)
		
		self.sampler = tf.identity(self.g_net(self.seq, self.z, train=False), name='sampler') 

		# TODO 
		"""
			r img, r text -> 1
			f img, r text -> 0
			r img, w text -> 0
			w img, r text -> 0
		"""
		# self.d = self.d_net(r_seq, r_img, reuse=False) 	# r img, r text
		# self.d_1 = self.d_net(r_seq, self.f_img) 		# f img, r text
		# self.d_2 = self.d_net(self.w_seq, self.img)		# r img, w text
		# self.d_3 = self.d_net(r_seq, self.w_img)		# w img, r text

		# epsilon = tf.random_uniform([], 0.0, 1.0)
		# img_hat = epsilon * r_img + (1 - epsilon) * self.f_img
		# d_hat = self.d_net(r_seq, img_hat)

		# ddx = tf.gradients(d_hat, img_hat)[0]
		# ddx = tf.reshape(ddx, [-1, self.img_row * self.img_col * 3])
		# ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
		# ddx = tf.reduce_mean(tf.square(ddx - 1.0) * self.alpha)
		
		# self.g_loss = -tf.reduce_mean(self.d_1)
		# self.d_loss = tf.reduce_mean(self.d) - (tf.reduce_mean(self.d_1)+tf.reduce_mean(self.d_2)+tf.reduce_mean(self.d_3))/3.
		# self.d_loss = -(self.d_loss - ddx)
		# =======================================================

		#train with real
		pred_real, pred_real_tag = self.d_net(self.seq, self.img, reuse=False)
		loss_d_real_label = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_real, labels=tf.ones_like(pred_real)))
		loss_d_real_tag = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_real_tag, labels=self.seq),axis=1))
		loss_d_real = self.la * loss_d_real_label + loss_d_real_tag

		#train with fake
		d_f_tag = self.w_seq
		fake = self.g_net(d_f_tag, self.z, reuse=True)
		pred_fake, pred_fake_tag = self.d_net(d_f_tag, fake)

		loss_d_fake_label = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_fake, labels=tf.zeros_like(pred_fake)))
		loss_d_fake_tag = tf.reduce_mean(tf.reduce_sum ( tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_fake_tag, labels=d_f_tag), axis=1))
		loss_d_fake = self.la * loss_d_fake_label #+ loss_d_fake_tag

		self.d_loss = loss_d_real + loss_d_fake 

		#update generator
		tags = self.w_seq
		self.generated = self.g_net(tags, self.z, reuse=True)

		d_g, d_gc = self.d_net(tags, self.generated)
		loss_g_label = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_g, labels=tf.ones_like(d_g)))
		loss_g_tag = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_gc, labels=tags), axis=1))

		self.g_loss = self.la * loss_g_label + loss_g_tag


		# =======================================================
		# dcgan
		# self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_1, labels=tf.ones_like(self.d_1))) 

		# self.d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d, labels=tf.ones_like(self.d))) \
		# 			+ (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_1, labels=tf.zeros_like(self.d_1))) + \
		# 			   tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_2, labels=tf.zeros_like(self.d_2))) +\
		# 			   tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_3, labels=tf.zeros_like(self.d_3))) ) / 3 
		

		self.global_step = tf.Variable(0, name='g_global_step', trainable=False)

		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.d_updates = tf.train.AdamOptimizer(self.FLAGS.lr, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=self.d_net.vars)
			self.g_updates = tf.train.AdamOptimizer(self.FLAGS.lr, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=self.g_net.vars, global_step=self.global_step)

		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

	def load_model(self, model_path):
		print("Restore Model...\n");
		self.saver.restore(self.sess, model_path)

	def train(self):
		batch_num = self.data.length//self.FLAGS.batch_size if self.data.length%self.FLAGS.batch_size==0 else self.data.length//self.FLAGS.batch_size + 1

		print("Start training ACGAN...\n")

		for t in range(self.FLAGS.iter):

			d_cost = 0
			g_coat = 0

			for d_ep in range(self.d_epoch):

				img, tags, _, w_img, w_tags = self.data.next_data_batch(self.FLAGS.batch_size)
				z = self.data.next_noise_batch(len(tags), self.FLAGS.z_dim)

				feed_dict = {
					self.seq:tags,
					self.img:img,
					self.z:z,
					self.w_seq:w_tags,
					self.w_img:w_img
				}

				_, loss = self.sess.run([self.d_updates, self.d_loss], feed_dict=feed_dict)

				d_cost += loss/self.d_epoch

			z = self.data.next_noise_batch(len(tags), self.FLAGS.z_dim)
			feed_dict = {
				self.img:img,
				self.w_seq:w_tags,
				self.w_img:w_img,
				self.seq:tags,
				self.z:z
			}

			_, loss, step = self.sess.run([self.g_updates, self.g_loss, self.global_step], feed_dict=feed_dict)

			current_step = tf.train.global_step(self.sess, self.global_step)

			g_cost = loss
			if current_step % self.FLAGS.display_every == 0:
				print("Epoch {}, Current_step {}".format(self.data.epoch, current_step))
				print("Discriminator loss :{}".format(d_cost))
				print("Generator loss     :{}".format(g_cost))
				print("---------------------------------")

			if current_step % self.FLAGS.checkpoint_every == 0:
				path = self.saver.save(self.sess, self.checkpoint_prefix, global_step=current_step)
				print ("\nSaved model checkpoint to {}\n".format(path))

			if current_step % self.FLAGS.dump_every == 0:
				self.eval(current_step)
				print("Dump test image")

	def eval(self, iters):
		z = self.data.fixed_z
		feed_dict = {
			self.seq:self.data.test_tags_idx,
			self.z:z
		}

		f_imgs = self.sess.run(self.sampler, feed_dict=feed_dict)

		data_utils.dump_img(self.FLAGS.img_dir, f_imgs, iters)
	
	def test(self):
		print("Start testing ACGAN...\n")
		# random_seeds = [1234, 1029+get_continue_sum(6), 840506+get_continue_sum(9), 12345+get_continue_sum(1), 666666+get_continue_sum(3), 10101+get_continue_sum(2), 6+get_continue_sum(3), 6+get_continue_sum(5), 6666+get_continue_sum(6),
		# 6666+get_continue_sum(3), 666+get_continue_sum(6), 666666+get_continue_sum(8), 77777+get_continue_sum(8), 8888+get_continue_sum(4), 88888+get_continue_sum(0), 88888+get_continue_sum(2), 22222+get_continue_sum(8), 22222+get_continue_sum(1),
		# 1111+get_continue_sum(7)]
		random_seeds = [687, 1050, 10104, 12346, 88888]
		for t in range(len(random_seeds)):
			print("test iteration: %d" % (t))
			random_seed = random_seeds[t]
			z = self.data.next_noise_batch(len(self.data.test_tags_idx), self.FLAGS.z_dim, random_state=random_seed)
			
			feed_dict = {
				self.seq:self.data.test_tags_idx,
				self.z:z
			}
			
			f_imgs = self.sess.run(self.sampler, feed_dict=feed_dict)
			data_utils.dump_test_img(self.FLAGS.samples_dir, f_imgs, t)



