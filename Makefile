default: build

DEEPSPEECH_RELEASE := 0.9.3
TECHIAITH_RELEASE := 21.03


run: 
	docker run --gpus all --name techiaith-deepspeech-train-v${DEEPSPEECH_RELEASE}-${USER} -it \
		-v ${PWD}/data/:/data \
		-v ${PWD}/checkpoints/:/checkpoints \
		-v ${PWD}/models/:/models \
		-v ${PWD}/export/:/export \
		-v ${PWD}/homedir/:/root \
		-v ${PWD}/local/:/DeepSpeech/bin/bangor_welsh \
		--env DEEPSPEECH_RELEASE=${DEEPSPEECH_RELEASE} \
		--env TECHIAITH_RELEASE=${TECHIAITH_RELEASE} \
		techiaith/deepspeech-train:v${DEEPSPEECH_RELEASE} bash


build:
	
	if [ ! -d "checkpoints/mozilla" ]; then \
	    mkdir -p checkpoints/mozilla; \
	    cd checkpoints/mozilla && \
		wget https://github.com/mozilla/DeepSpeech/releases/download/v$(DEEPSPEECH_RELEASE)/deepspeech-$(DEEPSPEECH_RELEASE)-checkpoint.tar.gz && \
		tar xvfz deepspeech-$(DEEPSPEECH_RELEASE)-checkpoint.tar.gz && \
		mv deepspeech-$(DEEPSPEECH_RELEASE)-checkpoint deepspeech-en-checkpoint;\
	fi
	if [ ! -d "checkpoints/techiaith" ]; then \
	    mkdir -p checkpoints/techiaith; \
	    cd checkpoints/techiaith && \
		wget https://github.com/techiaith/docker-deepspeech-cy/releases/download/$(TECHIAITH_RELEASE)/techiaith_bangor_$(TECHIAITH_RELEASE)_checkpoint.tar.gz && \
		tar zxvf techiaith_bangor_$(TECHIAITH_RELEASE)_checkpoint.tar.gz; \
	fi
	if [ ! -d "models/mozilla" ]; then \
	    mkdir -p models/mozilla; \
	    cd models/mozilla && \
		wget https://github.com/mozilla/DeepSpeech/releases/download/v$(DEEPSPEECH_RELEASE)/deepspeech-$(DEEPSPEECH_RELEASE)-models.pbmm && \
		wget https://github.com/mozilla/DeepSpeech/releases/download/v$(DEEPSPEECH_RELEASE)/deepspeech-$(DEEPSPEECH_RELEASE)-models.scorer;\
	fi
	if [ ! -d "models/techiaith" ]; then \
	    mkdir -p models/techiaith; \
	    cd models/techiaith && \
		wget https://github.com/techiaith/docker-deepspeech-cy/releases/download/$(TECHIAITH_RELEASE)/techiaith_bangor_$(TECHIAITH_RELEASE).pbmm && \
		wget https://github.com/techiaith/docker-deepspeech-cy/releases/download/$(TECHIAITH_RELEASE)/techiaith_bangor_macsen_$(TECHIAITH_RELEASE).scorer && \
		wget https://github.com/techiaith/docker-deepspeech-cy/releases/download/$(TECHIAITH_RELEASE)/techiaith_bangor_transcribe_$(TECHIAITH_RELEASE).scorer;\
	fi
	docker build --build-arg BRANCH=v${DEEPSPEECH_RELEASE} --rm -t techiaith/deepspeech-train:v${DEEPSPEECH_RELEASE} .


clean:
	-docker rmi techiaith/deepspeech-train:v${DEEPSPEECH_RELEASE}
	-docker rmi mozilla/deepspeech-train:v${DEEPSPEECH_RELEASE}	
	sudo rm -rf homedir
	sudo rm -rf checkpoints
	sudo rm -rf models


stop:
	-docker stop techiaith-deepspeech-train-v${DEEPSPEECH_RELEASE}-${USER}
	-docker rm techiaith-deepspeech-train-v${DEEPSPEECH_RELEASE}-${USER}
