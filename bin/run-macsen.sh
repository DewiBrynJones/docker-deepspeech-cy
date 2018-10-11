#!/bin/bash

if [ -n "${SINGULARITY_CONTAINER}" ]; then
	cd /DeepSpeech || exit
fi

checkpoint_dir=$(python -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/paldaruo"))')

python -u DeepSpeech.py \
	--train_files /data/paldaruo/deepspeech.csv \
	--dev_files /data/testsets/macsen/deepspeech.csv \
	--test_files /data/testsets/macsen/deepspeech.csv \
	--alphabet_config_path /data/paldaruo/alphabet.txt \
	--lm_binary_path /data/testsets/macsen/lm.binary \
	--lm_trie_path /data/testsets/macsen/trie \
	--validation_step 5 \
	--train_batch_size 12 \
	--dev_batch_size 12 \
	--test_batch_size 12 \
	--learning_rate 0.0001 \
	--epoch 45 \
	--display_step 5 \
	--validation_step 5 \
	--dropout_rate 0.20 \
	--default_stddev 0.046875 \
	--checkpoint_dir "$checkpoint_dir" \
	--export_dir /data/output \
	"$@"
