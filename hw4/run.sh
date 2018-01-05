if ! [ -f "./model/model-39500.data-00000-of-00001" ]; then
	wget -O ./model/model-39500.data-00000-of-00001 "https://www.dropbox.com/s/pcblcj2ehx9vg1m/model-39500.data-00000-of-00001?dl=1"
fi

if ! [ -f "./prepro/img_feat.dat" ]; then
	wget -O ./prepro/img_feat.dat "https://www.dropbox.com/s/lf24lskd7ofhy2x/img_feat.dat?dl=1"
fi

python generate.py --test_path $1