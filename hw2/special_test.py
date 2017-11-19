import model_seq2seq
import sys

data_dir = sys.argv[1];
output_file = sys.argv[2];
model_path='./models/model-90';
peer_output_file = 'tmp_peer_result.txt'
model_seq2seq.test(data_dir, output_file, peer_output_file, model_path, special_test=True);