import model_seq2seq
import sys

if(len(sys.argv) == 4):
	data_dir = sys.argv[1];
	test_output_file = sys.argv[2];
	peer_review_output_file = sys.argv[3];
else:
	print('Error input format, stop testing');
	exit();

# ====== path setting =======
# data_dir = './MLDS_hw2_data';
# test_output_file = 'S2VT_attention_results.txt';
model_path='./models/model-90';

model_seq2seq.test(data_dir, test_output_file, peer_review_output_file, model_path);