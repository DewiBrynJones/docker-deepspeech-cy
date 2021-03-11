#!/bin/bash
set -e
set -u
set -o pipefail

help()
{
	echo
	echo "Rhedeg sgriptiau optimeiddio modelau iaith KenLM"
	echo "Run scripts for optimizing KenLM language models"
	echo
	echo "Usage: $ `basename $0` [OPTIONS]"
	echo
	echo "Options:"
	echo
	echo " -c, --csv_test_file        Path csv file containing transcriptions with paths to clips"
	echo " -d, --domain               Name for language model domain"
	echo " -p, --checkpoint_dir       Path to checkpoint directory (optional)"
	echo
	exit 0
}

lm_domain=''
test_file=''
checkpoint_cy_dir=''

SHORT=hd:c:p
LONG=help,domain:,csv_test_file:,checkpoint_dir

# read options
OPTS=$(getopt --options $SHORT --long $LONG --name "$0" -- "$@")

if [ $? != 0 ] ; then 
	echo "Failed to parse options...exiting." >&2 ; 
	exit 1 ; 
fi

eval set -- "$OPTS"

while true ; do
  case "$1" in
    -d | --domain )
      lm_domain="$2"
      shift 2
      ;;
    -t | --csv_test_file )
      test_file="$2"
      shift 2
      ;;
    -p | --checkpoint_dir )
      checkpoint_cy_dir="$2"
      shift 2
      ;;
    -h | --help )
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

if [ -z "${test_file}" ]; then
    echo "--csv_test_file missing. Use `basename $0` -h for more info." 
    exit 2
fi

if [ -z "${lm_domain}" ]; then
    echo "--domain missing. Use `basename $0` -h for more info."
    exit 2 
fi

if [ -z "$checkpoint_cy_dir" ]; then
    checkpoint_cy_dir=/checkpoints/cy
    echo "-p|--checkpoint_dir not set. Setting to ${checkpoint_cy_dir} "
fi


# If checkpoint dir is empty, copy pretrained from techiaith..
pretrained_checkpoint_dir=/checkpoints/techiaith
if [ ! "$(ls -A ${checkpoint_cy_dir})" ]; then
  cp -r ${pretrained_checkpoint_dir} ${checkpoint_cy_dir}
fi


VOCAB_SIZE=50000
alphabet_file_path=/DeepSpeech/bin/bangor_welsh/alphabet.txt
output_dir=/export/${DEEPSPEECH_RELEASE}_${TECHIAITH_RELEASE}


cd ${output_dir}


# Force UTF-8 output
export PYTHONIOENCODING=utf-8	

echo "####################################################################################"
echo "#### Determine optimal alpha and beta parameters                                ####"
echo "####################################################################################"
python /DeepSpeech/lm_optimizer.py \
  --test_files ${test_file} \
  --checkpoint_dir ${checkpoint_cy_dir} \
  --alphabet_config_path ${alphabet_file_path} \
  --scorer kenlm.${lm_domain}.scorer


echo "####################################################################################"
echo "#### Input required....                                                         ####"
echo "####################################################################################"
read -p "Enter best default alpha: " alpha
read -p "Enter best default beta: " beta


echo "####################################################################################"
echo "#### saving optimal alpha and beta values. run build_lm_scorer.sh once more     ####"
echo "####################################################################################"
default_alpha_file="${output_dir}/optimal_alpha.${lm_domain}.txt"
default_beta_file="${output_dir}/optimal_beta.${lm_domain}.txt"
echo ${alpha} > ${default_alpha_file}
echo ${beta} > ${default_beta_file}

cd -
