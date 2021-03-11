#!/bin/bash
set -e

help()
{
	echo
	echo "Rhedeg sgriptiau hyfforddi modelau acwstig DeepSpeech gyda setiau data Mozilla o CommonVoice"
	echo "Run scripts for training DeepSpeech acoustic models with Mozilla prescribed datasets from CommonVoice"
	echo
	echo "Syntax: $ `basename $0` [OPTIONS]"
	echo
	echo "Options:"
	echo
	echo " -c, --cv_dir         Path to CommonVoice data directory "
	echo "                      (imported with import_cv_archive.py) "
	echo
	exit 0
}


SHORT=hc:
LONG=cv_dir:

cv_dir=''


# read options
OPTS=$(getopt --options $SHORT --long $LONG --name "$0" -- "$@")

if [ $? != 0 ] ; then 
	echo "Failed to parse options...exiting." >&2 ; 
	exit 1 ; 
fi

eval set "$OPTS"

while true ; do
  case "$1" in
    -c | --cv_dir )
      cv_dir="$2"
      shift 2
      ;;
    -h)
      help
      shift
      ;;    
    *)
      help
      exit 1
      ;;
  esac
done


###
model_name='bangor-mozilla-welsh'
model_language='cy-Latn-GB'
model_license='MPL'
model_description='Welsh language acoustic model trained using transfer learning and Mozilla''s prescribed CommonVoice datasets for training, validation and testing.'
model_author='techiaith'
model_contact_info='techiaith@bangor.ac.uk'

echo
echo "####################################################################################"
echo " model_name : ${model_name}"
echo " model_language : ${cy-Latn-GB}"
echo " model_license : ${model_license}"
echo " model_description : ${model_description}"
echo " model_author : ${model_author}"
echo " model_contact_info : ${model_contact_info}"
echo " model_version : ${TECHIAITH_RELEASE} "
echo " DeepSpeech Version : ${DEEPSPEECH_RELEASE} "
echo "####################################################################################"
echo

###
train_files=${csv_dir}/train.csv
devset_files=${csv_dir}/dev.csv
test_files=${csv_dir}/test.csv

alphabet_cy_file=/DeepSpeech/bin/bangor_welsh/alphabet.txt

checkpoint_dir=/checkpoints
export_dir=/export/${DEEPSPEECH_RELEASE}_${TECHIAITH_RELEASE}


### Force UTF-8 output
export PYTHONIOENCODING=utf-8

checkpoint_en_dir="${checkpoint_dir}/en"
checkpoint_cy_dir="${checkpoint_dir}/cy-moz"

rm -rf ${checkpoint_en_dir}
rm -rf ${checkpoint_cy_dir}
rm -rf ${export_dir}

mkdir -p ${checkpoint_en_dir}
mkdir -p ${checkpoint_cy_dir}
mkdir -p ${export_dir}

cp -r /checkpoints/mozilla/deepspeech-en-checkpoint/ $checkpoint_en_dir

###
echo
echo "####################################################################################"
echo "#### Transfer to WELSH model with --save_checkpoint_dir --load_checkpoint_dir   ####"
echo "####################################################################################"
set -x
python -u DeepSpeech.py \
	--train_files "${train_files}" \
	--dev_files "${devset_files}" \
	--train_batch_size 24 \
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
	--train_files "${train_files}" \
	--train_batch_size 64 \
	--test_files "${test_files}" \
	--epochs 1 \
	--alphabet_config_path "${alphabet_cy_file}" \
	--save_checkpoint_dir "${checkpoint_cy_dir}" \
	--load_checkpoint_dir "${checkpoint_cy_dir}" \
	--remove_export \
	--export_dir "${export_dir}" \
	--export_author_id "${model_author}" \
	--export_model_name "${model_name}" \
	--export_model_version "${TECHIAITH_RELEASE}" \
	--export_contact_info "${model_contact_info}" \
	--export_license "${model_license}" \
	--export_language "${model_language}" \
	--export_min_ds_version "${DEEPSPEECH_RELEASE}" \
	--export_max_ds_version "${DEEPSPEECH_RELEASE}" \
	--export_description "${model_description}"

###
/DeepSpeech/native_client/convert_graphdef_memmapped_format \
	--in_graph=${export_dir}/output_graph.pb \
	--out_graph=${export_dir}/output_graph.pbmm


set +x
echo
echo "####################################################################################"
echo "#### Exported acoustic models (.pb/.pbmm files) can be found in ${export_dir} "
echo "####################################################################################"
set -x
