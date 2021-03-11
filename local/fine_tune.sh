#!/bin/bash
set -e
set -u
set -o pipefail

help()
{
	echo
	echo "Rhedeg sgriptiau creu modelau wedi eu gwella gyda data ychwanegol"
	echo "Run scripts for fine tuning models with new additional data"
	echo
	echo "Usage: $ `basename $0` [OPTIONS]"
	echo
	echo "Options:"
	echo
	echo " -c, --csv_train_file       Path to csv file containing extra transcriptions with paths to audio clips"
  echo " -p, --checkpoint_dir       Path to previously trained checkpoint directory (optional - default /checkpoints/cy)"
	echo " -t, --text_file            Path to text file containing union of all corpora (e.g. corpus.union.clean.txt"
	echo " -n, --name                 Name for fine tuning"
	echo " -d, --domain               language model domain (e.g. 'macsen' or 'transcribe')"
	echo
	exit 0
}

csv_file=''
lm_domain=''
fine_tune_name=''
source_text_file=''
pretrained_checkpoint_dir=''

SHORT=hc:t:n:p:d:
LONG=csv_train_file:,text_file:,name:,checkpoint_dir:,domain:

# read options
OPTS=$(getopt --options $SHORT --long $LONG --name "$0" -- "$@")

if [ $? != 0 ] ; then 
	echo "Failed to parse options...exiting." >&2 ; 
	exit 1 ; 
fi

eval set -- "$OPTS"

while true ; do
  case "$1" in
  	-c | --csv_train_file )
      csv_file="$2"
      shift 2
      ;;
    -t | --text_file )
      source_text_file="$2"
      shift 2
      ;;
    -n | --name )
      fine_tune_name="$2"
      shift 2
      ;;
	-d | --domain )
      lm_domain="$2"
      shift 2
      ;;
	-p | --checkpoint_dir )
      pretrained_checkpoint_dir="$2"
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


if [ -z "${csv_file}" ]; then
    echo "--csv_train_file missing. Use `basename $0` -h for more info."                                                                               
    exit 2                                                                                                                                        
fi

if [ -z "${source_text_file}" ]; then
    echo "--text_file missing. Use `basename $0` -h for more info."                                                                               
    exit 2                                                                                                                                        
fi

if [ -z "${lm_domain}" ]; then
    echo "--domain missing. Use `basename $0` -h for more info."
    exit 2
fi

if [ -z "${fine_tune_name}" ]; then
    echo "--name missing. Use `basename $0` -h for more info."
    exit 2
fi


if [ -z "${pretrained_checkpoint_dir}" ]; then
	checkpoint_cy_dir=/checkpoints/cy
	pretrained_checkpoint_dir=/checkpoints/techiaith
	if [ "$(ls -A ${checkpoint_cy_dir})" ]; then
		pretrained_checkpoint_dir=${checkpoint_cy_dir}		
	fi	
fi

#
export_dir=/export/${DEEPSPEECH_RELEASE}_${TECHIAITH_RELEASE}/${fine_tune_name}
checkpoint_finetuned_dir=/checkpoints/${fine_tune_name}

rm -rf ${export_dir}
rm -rf ${checkpoint_finetuned_dir}

mkdir -p ${export_dir}
mkdir -p ${checkpoint_finetuned_dir}


###
train_files=${csv_file}
alphabet_cy_file=/DeepSpeech/bin/bangor_welsh/alphabet.txt

### Force UTF-8 output
export PYTHONIOENCODING=utf-8

set +x
echo
echo "####################################################################################"
echo "#### Continue acoustic model training from best previous checkpoint             ####"
echo "####################################################################################"
set -x
python -u DeepSpeech.py \
	--train_files "${train_files}" \
	--train_batch_size 2 \
	--epochs 10 \
	--alphabet_config_path "${alphabet_cy_file}" \
	--load_checkpoint_dir "${pretrained_checkpoint_dir}" \
	--save_checkpoint_dir "${checkpoint_finetuned_dir}" \
	--export_dir "${export_dir}"

###
/DeepSpeech/native_client/convert_graphdef_memmapped_format \
	--in_graph=${export_dir}/output_graph.pb \
	--out_graph=${export_dir}/output_graph.pbmm


set +x
echo
echo "####################################################################################"
echo "#### Fine tuned acoustic models (.pb/.pbmm files) can be found at ${export_dir} "
echo "####################################################################################"
set -x

set +x
echo "####################################################################################"
echo "#### Generating finetuned binary language model                                 ####"
echo "####################################################################################"
set -x

/DeepSpeech/bin/bangor_welsh/build_lm_scorer.sh \
	--text_file ${source_text_file} \
	--domain ${lm_domain} \
	--output_dir ${export_dir}
