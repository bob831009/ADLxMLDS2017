import S2VT_model
import sys

data_dir = sys.argv[1];
output_file = sys.argv[2];
model_path='./models/model-990';
S2VT_model.test(data_dir, output_file, model_path, special_test=True);