#!/bin/bash
set -e
set -u
set -o pipefail

help()
{
	echo
	echo "Rhedeg sgriptiau hyfforddi modelau iaith KenLM i'w defnyddio gyda DeepSpeech"
	echo "Run scripts for training KenLM language models for use with DeepSpeech"
	echo
	echo "Syntax: $ `basename $0` [OPTIONS]"
	echo
	echo "Options:"
	echo
	echo " -t, --text_file        Path to text file containing all corpus text "
	echo " -d, --domain           Name for language model domain (e.g. 'macsen' or 'transcribe' "
  echo " -o, --output_dir       (optional) Default: /export/${DEEPSPEECH_RELEASE}_${TECHIAITH_RELEASE}"
	echo
	exit 0
}

lm_domain=''
source_text_file=''
output_dir=/export/${DEEPSPEECH_RELEASE}_${TECHIAITH_RELEASE}

SHORT=ht:d:o:
LONG=text_file:,domain:,output_dir:

# read options
OPTS=$(getopt --options $SHORT --long $LONG --name "$0" -- "$@")


if [ $? != 0 ] ; then 
	echo "Failed to parse options...exiting." >&2 ; 
	exit 1 ; 
fi

eval set -- "$OPTS"

while true ; do
  case "$1" in
    -t | --text_file )
      source_text_file="$2"
      shift 2
      ;;
    -d | --domain )
      lm_domain="$2"
      shift 2
      ;;
    -o | --output_dir )
      output_dir="$2"
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

if [ -z "${source_text_file}" ]; then
    echo "--text_file missing. Use `basename $0` -h for more info."                                                                               
    exit 2                                                                                                                                        
fi

if [ -z "${lm_domain}" ]; then
    echo "--domain missing. Use `basename $0` -h for more info."
    exit 2
fi

mkdir -p ${output_dir}
cd ${output_dir}

VOCAB_SIZE=50000
alphabet_file_path=/DeepSpeech/bin/bangor_welsh/alphabet.txt



set +x
echo "####################################################################################"
echo "#### Generating binary language model                                           ####"
echo "####################################################################################"
set -x
python /DeepSpeech/data/lm/generate_lm.py \
  --input_txt "${source_text_file}" \
  --output_dir . \
  --top_k ${VOCAB_SIZE} \
  --kenlm_bins '/DeepSpeech/native_client/kenlm/build/bin/' \
  --arpa_order 6 \
  --max_arpa_memory '85%' \
  --arpa_prune "0|0|1" \
  --binary_a_bits 255 \
  --binary_q_bits 8 \
  --binary_type 'trie' \
  --discount_fallback

#
set +x

default_alpha=1.7242448485503816
default_beta=4.9065413926676165

default_alpha_file="${output_dir}/optimal_alpha.${lm_domain}.txt"
default_beta_file="${output_dir}/optimal_beta.${lm_domain}.txt"

bangor_default_alpha_file=/DeepSpeech/bin/bangor_welsh/conf/default_lm_alpha.${lm_domain}.txt
bangor_default_beta_file=/DeepSpeech/bin/bangor_welsh/conf/default_lm_beta.${lm_domain}.txt

if [ -f ${bangor_default_alpha_file} ] ; then
  if [ ! -f ${default_alpha_file} ] ; then
    cp ${bangor_default_alpha_file} ${default_alpha_file}
  fi
fi

if [ -f ${bangor_default_beta_file} ] ; then
  if [ ! -f ${default_beta_file} ] ; then    
    cp ${bangor_default_beta_file} ${default_beta_file}
  fi
fi

if [ -f ${default_alpha_file} ] ; then
  default_alpha=$(<${default_alpha_file})
fi

if [ -f ${default_beta_file} ] ; then
  default_beta=$(<${default_beta_file})
fi


set +x
echo "####################################################################################"
echo "#### Generating language model package                                          ####"
echo "####                                                                            ####"
echo "#### Default alpha and beta values are                                          ####"
echo "####                                                                            ####"
echo "####  alpha : ${default_alpha}                                                ####"
echo "####  beta  : ${default_beta}                                                ####"
echo "####                                                                            ####"
echo "####################################################################################"
set -x

/DeepSpeech/native_client/generate_scorer_package \
	--alphabet "${alphabet_file_path}" \
	--lm lm.binary \
	--vocab vocab-${VOCAB_SIZE}.txt \
	--package kenlm.${lm_domain}.scorer \
 	--default_alpha ${default_alpha} \
	--default_beta ${default_beta}

cd -

set +x
echo "####################################################################################"
echo "#### Successfully built lm package : ${output_dir}/kenlm.${lm_domain}.scorer "
echo "####################################################################################"
set -x
