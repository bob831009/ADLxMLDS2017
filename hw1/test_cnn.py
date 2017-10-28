from __future__ import print_function
from keras.models import load_model
import numpy as np
import sys
import os

# ========== parameter ===========
data_dir = sys.argv[1];
output_file = sys.argv[2];
model_dir = './model/cnn1_bidirLstm6.h5';
# ================================

source_test_x = {};
test_sentence_names = [];
phone2num = {};
phone2char = {};
phone2phone = {};
num2phone = {};
max_frame_num = 800;
label_num = 49;
feature_dim = 39;

def getSentence(labels, sentence_len):
	find_first_phone = False;
	org_sentence = '';
	sentence = '';
	pre_phone = '';
	for i in range(sentence_len):
		lab = labels[i];
		phone = phone2phone[num2phone[lab]];
		org_sentence += phone2char[phone];
		if(phone != 'sil'):
			find_first_phone = True;
		if(phone == 'sil' and (not find_first_phone)):
			continue;
		if(phone == pre_phone):
			continue;
		sentence += phone2char[phone];
		pre_phone = phone;
	if(sentence[-1] == 'L'):
		sentence = sentence[:-1];
	return sentence;



# ======== load data ===========
print('loading data...');

with open(os.path.join(data_dir, 'mfcc/test.ark')) as f:
	for line in f:
		line = line.strip().split(' ');
		sentence_name = line[0].rsplit('_', 1)[0];
		if(sentence_name not in source_test_x):
			source_test_x[sentence_name] = [];
		source_test_x[sentence_name].append(np.array(line[1:]));

with open(os.path.join(data_dir, '48phone_char.map')) as f:
	for line in f:
		line = line.strip().split('\t');
		phone2num[line[0]] = int(line[1]);
		phone2char[line[0]] = line[2];
		num2phone[int(line[1])] = line[0]
num2phone[label_num - 1] = 'sil';

with open(os.path.join(data_dir, 'phones/48_39.map')) as f:
	for line in f:
		line = line.strip().split('\t');
		phone2phone[line[0]] = line[1];

test_sentence_names = source_test_x.keys();

sentence_num = len(test_sentence_names);
test_x = np.zeros((sentence_num, max_frame_num, feature_dim));
for i, key in enumerate(test_sentence_names):
	for j in range(len(source_test_x[key])):
		for k in range(feature_dim):
			test_x[i][j][k] = source_test_x[key][j][k];

# ========== load model and test ===========
print('loading model...');
model = load_model(model_dir);

print('start testing...');
predict_y = model.predict_classes(test_x);

print('output to file...')
with open(output_file, 'w') as f:
	f.write('id,phone_sequence\n');
	for i, key in enumerate(test_sentence_names):
		f.write('%s,%s\n' % (key, getSentence(predict_y[i], len(source_test_x[key]) )));

