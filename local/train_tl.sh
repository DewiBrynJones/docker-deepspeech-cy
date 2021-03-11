#!/bin/bash
set -e

help()
{
	echo
	echo "Rhedeg sgriptiau hyfforddi modelau acwstig DeepSpeech gyda data o CommonVoice"
	echo "Run scripts for training DeepSpeech acoustic models with data from CommonVoice"
	echo
	echo "Syntax: $ `basename $0` [OPTIONS]"
	echo
	echo "Options:"
	echo
	echo " -c, --csv_files                  One or more CSV files to be used for training"
	echo " -s, --save_checkpoint_dir        Path to directory for saving checkpoints (Optional. Default /checkoints/cy)"
	echo
	exit 0
}


SHORT=hc:s:
LONG=help,csv_files:,save_checkpoint_dir:

csv_files=''
save_checkpoint_dir=''

# read options
OPTS=$(getopt --options $SHORT --long $LONG --name "$0" -- "$@")


if [ $? != 0 ] ; then 
	echo "Failed to parse options...exiting." >&2 ; 
	exit 1 ; 
fi

eval set -- "$OPTS"

while true ; do
  case "$1" in
    -c | --csv_files )
      csv_files="$2"
      shift 2
      ;;
	-s | --save_checkpoint_dir )
	  save_checkpoint_dir="$2"
	  shift 2
	  ;;
    -h | --help)
      help
      shift
      ;;
	-- )
	  shift
	  break
	  ;;
    *)
      help
      exit 1
      ;;
  esac
done

#
if [ -z "${csv_files}" ]; then
    echo "--csv_files missing. Use `basename $0` -h for more info."	
    exit 2
fi


###
checkpoint_dir=/checkpoints
checkpoint_en_dir="${checkpoint_dir}/en"
checkpoint_cy_dir=${save_checkpoint_dir}
if [ -z "${save_checkpoint_dir}" ]; then
    checkpoint_cy_dir="${checkpoint_dir}/cy"
fi


###
alphabet_cy_file=/DeepSpeech/bin/bangor_welsh/alphabet.txt


### Force UTF-8 output
export PYTHONIOENCODING=utf-8

rm -rf ${checkpoint_en_dir}
rm -rf ${checkpoint_cy_dir}

mkdir -p ${checkpoint_en_dir}
mkdir -p ${checkpoint_cy_dir}

cp -r /checkpoints/mozilla/deepspeech-en-checkpoint/ $checkpoint_en_dir

###
echo
echo "####################################################################################"
echo "#### Transfer to WELSH model with --save_checkpoint_dir --load_checkpoint_dir   ####"
echo "####################################################################################"
set -x
python -u DeepSpeech.py \
	--train_files "${csv_files}" \
	--train_batch_size 64 \
	--drop_source_layers 2 \
	--epochs 10 \
	--alphabet_config_path "${alphabet_cy_file}" \
	--load_checkpoint_dir "${checkpoint_en_dir}" \
	--save_checkpoint_dir "${checkpoint_cy_dir}"
	

set +x
echo
echo "####################################################################################"
echo "#### Export new Welsh checkpoint to frozen model                                ####"
echo "####################################################################################"
set -x
python -u DeepSpeech.py \
	--train_files "${train_files}" --train_batch_size 64 \
	--epochs 1 \
	--alphabet_config_path "${alphabet_cy_file}" \
	--load_checkpoint_dir "${checkpoint_cy_dir}" \
	--save_checkpoint_dir "${checkpoint_cy_dir}"	
