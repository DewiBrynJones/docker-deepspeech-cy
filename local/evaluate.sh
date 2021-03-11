#!/bin/bash
set -e
set -u
set -o pipefail

help()
{
	echo
	echo "Rhedeg sgriptiau profi modelau DeepSpeech erbyn set profi benodol"
	echo "Run scripts for testing DeepSpeech models against a specific test set"
	echo
	echo "Usage: $ `basename $0` [OPTIONS]"
	echo
	echo "Options:"
	echo
	echo " -c, --csv_test_file        Path to test set csv file containing paths to clips and reference transcriptions"
  echo " -s, --scorer               Path to language model scorer"
  echo " -p, --checkpoint_dir       Path to checkpoint directory (optional)"
  echo " -r, --results_file         Path to results file (optional)"
	echo
	exit 0
}

scorer=''
test_file=''
results_file=''
checkpoint_cy_dir=''

SHORT=hs:c:p:r:
LONG=scorer:,csv_test_file:,checkpoint_dir:,results_file:

# read options
OPTS=$(getopt --options $SHORT --long $LONG --name "$0" -- "$@")


if [ $? != 0 ] ; then 
	echo "Failed to parse options...exiting." >&2 ; 
	exit 1 ; 
fi

eval set -- "$OPTS"

while true ; do
  case "$1" in
    -s | --scorer )
      scorer="$2"      
      shift 2
      ;;
    -c | --csv_test_file )
      test_file="$2"
      shift 2
      ;;
    -p | --checkpoint_dir )
      checkpoint_cy_dir="$2"
      shift 2
      ;;
    -r | --results_file )
      results_file="$2"
      shift 2
      ;;
    -h)
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

if [ -z "${scorer}" ]; then
    echo "--scorer missing. Use `basename $0` -h for more info." 
    exit 2
fi

if [ -z "$checkpoint_cy_dir" ]; then
    checkpoint_cy_dir=/checkpoints/cy
    echo "-p|--checkpoint_dir not set. Setting to  ${checkpoint_cy_dir} "
fi

if [ -z "$results_file" ]; then
    results_file=${test_file}.results.json    
fi

alphabet_file_path=/DeepSpeech/bin/bangor_welsh/alphabet.txt

set +x
echo "####################################################################################"
echo "#### evaluating with transcriber testset                											   ###"
echo "####################################################################################"
set -x

python -u /DeepSpeech/evaluate.py \
	--test_files "${test_file}" \
	--test_batch_size 1 \
	--alphabet_config_path "${alphabet_file_path}" \
	--load_checkpoint_dir "${checkpoint_cy_dir}" \
	--scorer_path ${scorer} \
  --test_output_file ${results_file}


set +x
echo "####################################################################################"
echo "#### Results in ${results_file} "
echo "####################################################################################"
set -x