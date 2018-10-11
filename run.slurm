#!/bin/bash
#SBATCH --output=cy_deepspeech.out.%J
#SBATCH --error=cy_deepspeech.err.%J
#SBATCH --job-name=cy_deepspeech
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --exclusive
#SBATCH -A scw1097

module purge
module load singularity

echo Running on host $(hostname)
echo Time is $(date)
echo SLURM job ID is ${SLURM_JOB_ID}
echo Singularity image ${DEEPSPEECH_IMG}
echo DeepSpeech script ${DEEPSPEECH_RUN}

start="$(date +%s)"

OUTPUT_DIR=/scratch/scw1097/cy_deepspeech/${SLURM_JOB_ID}
mkdir -vp ${OUTPUT_DIR}

{
	set -o pipefail
	# force-killing all sub-processes on process exit, Ctrl-C, kill-signal
	# "trap - SIGTERM" hinders "kill" from being killed itself
	trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

	compute_cmd="singularity exec --nv -B data,docker-deepspeech-cy/bin:/DeepSpeech/bin,/usr/local/cuda/lib64/stubs/ ${DEEPSPEECH_IMG} docker-deepspeech-cy/bin/${DEEPSPEECH_RUN}"
	srun="srun --exclusive -N1 -n1"

	# unfolding slurm's compact cluster representation
	nodes_raw=$(scontrol show hostname ${SLURM_JOB_NODELIST})
	index=0
	for node in $nodes_raw; do
		# keeping nodes as array for later index lookup
		nodes[$index]=$node
		((index = index + 1))
	done
	# comma separated node list (with leading comma)
	raw_node_list=$(printf ",%s" "${nodes[@]}")

	# exporting COMPUTE variables for the compute script that will
	# get executed on every node of the allocated cluster
	export COMPUTE_NODES=${raw_node_list:1}
	export COMPUTE_NODES_COUNT=${#nodes[@]}
	export COMPUTE_ID="paldaruo"
	export COMPUTE_NAME="paldaruo"
	export COMPUTE_JOB_NUMBER=${SLURM_JOB_ID}
	export COMPUTE_DATA_DIR=${OUTPUT_DIR}/data/scw_shared
	export COMPUTE_RESULTS_DIR=${OUTPUT_DIR}/results
	export COMPUTE_KEEP_DIR=${COMPUTE_RESULTS_DIR}/keep
	export COMPUTE_JOB_LOG="${OUTPUT_DIR}/paldaruo.log"
	export COMPUTE_GLOBAL_LOG="${OUTPUT_DIR}/global.log"

	for index in $(seq 0 $((COMPUTE_NODES_COUNT - 1))); do
		# will tell every instance of the .compute script, which node of the cluster it represents
		export COMPUTE_NODE_INDEX=$index
		# the node has to be specified by "-w" to guarantee execution under the correct COMPUTE_NODE_INDEX
		# if some log line contains "GLOBAL LOG", the remaining string is emitted to the provided log file
		$srun -w ${nodes[index]} /bin/bash -c "$compute_cmd" |
			tee >(sed -n -e 's/^.*GLOBAL LOG: //p' >>${COMPUTE_GLOBAL_LOG}) &
	done

	for index in $(seq 1 ${COMPUTE_NODES_COUNT}); do
		# "wait -n" waits for any sub-process to exit
		# doing this COMPUTE_NODES_COUNT times will wait for all node sub-processes to finish
		# in case of any node sub-process failing, it will exit immediately
		#wait -n
		code=$?
		if ((code > 0)); then
			echo "One node failed with exit code $code."
			exit $code
		else
			echo "One node succeeded."
		fi
	done

	echo "Success. Quitting..."

} 2>&1 | tee ${OUTPUT_DIR}/paldaruo.log

stop="$(date +%s)"
finish=$((${stop} - ${start}))
echo ${SLURM_JOB_ID} Job-Time ${finish} seconds
echo End Time is $(date)
