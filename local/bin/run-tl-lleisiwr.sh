#!/bin/bash

if [ -n "${SINGULARITY_CONTAINER}" ]; then
	cd /DeepSpeech || exit
fi

checkpoint_dir=$(python -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/gwion-tl-lleisiwr"))')

echo $checkpoint_dir


python -u /DeepSpeech/DeepSpeech.py \
	--train_files /data/corpws_lleisiwr/corpws_gwion/train_1.csv \
	--test_files /data/corpws_lleisiwr/corpws_gwion/test_2.csv \
	--alphabet_config_path /DeepSpeech/bin/bangor_welsh/alphabet.txt \
	--lm_binary_path /data/corpws_lleisiwr/corpws_gwion/lm.binary \
	--lm_trie_path /data/corpws_lleisiwr/corpws_gwion/trie \
        --source_model_checkpoint_dir /checkpoints/mozilla/deepspeech-0.5.1-checkpoint \
	--nofine_tune \
	--epochs 10 \
	--drop_source_layers 1 \
	--train_batch_size 48 \
	--test_batch_size 48 \
	--dev_batch_size 48 \
	--checkpoint_dir "$checkpoint_dir" \
	--summary_dir /keep/transfer/summaries \
	--export_dir /export/gwion-tl-lleisiwr \
	"$@"


#cp bin/bangor_welsh/alphabet.txt /export/CofnodYCynulliad

#cp /data/corpws_tts/trie /export/CofnodYCynulliad
#cp /data/corpws_tts/lm.binary /export/CofnodYCynulliad

#convert_graphdef_memmapped_format --in_graph=/export/macsen/output_graph.pb --out_graph=/export/macsen/output_graph.pbmm

#!/bin/bash

