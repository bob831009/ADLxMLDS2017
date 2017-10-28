from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, Bidirectional
from keras.layers import Conv1D
from keras.layers import TimeDistributed
from keras.callbacks import Callback, EarlyStopping
from keras.layers.core import Masking
import numpy as np
import random
import math

output_model_path = './model/cnn_rnn.h5';

source_train_x = {};
source_train_y = {};
phone2num = {};
phone2char = {};
max_frame_num = 800;
label_num = 49;
feature_dim = 39;

filters = 64;
kernel_size = 5;
lstm_output_size = 400;
batch_size = 64;
epochs = 20;


def load_data(x, y, key, index):
	for i in range(len(source_train_x[key])):
		for j in range(feature_dim):
			label = source_train_y[key][i];
			x[index][i][j] = source_train_x[key][i][j];
			y[index][i][label] = 1;
	for i in range(len(source_train_x[key]), max_frame_num):
		y[index][i][label_num - 1] = 1;

def evaluate(predict_y, valid_y):
	total = 0;
	correct = 0;
	for i in range(len(predict_y)):
		for j in range(len(predict_y[i])):
			if(valid_y[i][j][label_num-1] == 1):
				continue;
			total += 1;
			predict_label = predict_y[i][j];
			if(valid_y[i][j][predict_label] == 1):
				correct += 1;
	return float(correct)/total;

class TestCallback(Callback):
	def __init__(self, eval_data):
		self.eval_data = eval_data
		self.eval_acc = 0;

	def on_epoch_end(self, epoch, logs={}):
		x, y = self.eval_data
		pred_y = model.predict_classes(x);
		acc = evaluate(pred_y, y);

		if(acc > self.eval_acc):
			self.eval_acc = acc;
		else:
			if(acc > 0.75):
				self.model.stop_training = True;

		print('\nTesting loss: {}\n'.format(acc));

# ======== load data ===========
print('loading data...');
with open('./data/mfcc/train.ark') as f:
	for line in f:
		line = line.strip().split(' ');
		sentence_name = line[0].rsplit('_', 1)[0];
		if(sentence_name not in source_train_x):
			source_train_x[sentence_name] = [];
		source_train_x[sentence_name].append(np.array(line[1:]));

with open('./data/48phone_char.map') as f:
	for line in f:
		line = line.strip().split('\t');
		phone2num[line[0]] = int(line[1]);
		phone2char[line[0]] = line[2];

with open('./data/label/train.lab') as f:
	for line in f:
		line = line.strip().split(',');
		sentence_name = line[0].rsplit('_', 1)[0];
		if(sentence_name not in source_train_y):
			source_train_y[sentence_name] = [];
		source_train_y[sentence_name].append(phone2num[line[1]]);

sentence_num = len(source_train_y.keys());
valid_size = int(sentence_num / 10);
train_size = sentence_num - valid_size;

train_x = np.zeros((train_size, max_frame_num, feature_dim));
train_y = np.zeros((train_size, max_frame_num, label_num));

valid_x = np.zeros((valid_size, max_frame_num, feature_dim));
valid_y = np.zeros((valid_size, max_frame_num, label_num));

all_sentence = list(source_train_y.keys());
random.shuffle(all_sentence, random.random);

train_keys = all_sentence[:train_size];
valid_keys = all_sentence[train_size:];


for i, key in enumerate(train_keys):
	load_data(train_x, train_y, key, i);
for i, key in enumerate(valid_keys):
	load_data(valid_x, valid_y, key, i);

# ============ model setting ===============
model = Sequential()
model.add(Conv1D(filters,
                 kernel_size,
                 padding='same',
                 input_shape=train_x.shape[1:]
                 ))
model.add(Masking(mask_value=0.));
model.add(Bidirectional(LSTM(lstm_output_size, return_sequences=True)));
model.add(Bidirectional(LSTM(lstm_output_size, return_sequences=True)));
model.add(Bidirectional(LSTM(lstm_output_size, return_sequences=True)));
model.add(Bidirectional(LSTM(lstm_output_size, return_sequences=True)));
model.add(Bidirectional(LSTM(lstm_output_size, return_sequences=True)));
model.add(Bidirectional(LSTM(lstm_output_size, return_sequences=True)));
model.add(TimeDistributed(Dense(label_num, activation='softmax')));

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# ============ start training ==============
model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(valid_x, valid_y),
          callbacks=[EarlyStopping(monitor='val_loss', patience=2, verbose=0)]);
model.save(output_model_path);

predict_y = model.predict_classes(valid_x);
print('Test accuracy:', evaluate(predict_y, valid_y));
