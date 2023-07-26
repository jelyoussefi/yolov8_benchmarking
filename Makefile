#----------------------------------------------------------------------------------------------------------------------
# Flags
#----------------------------------------------------------------------------------------------------------------------
SHELL:=/bin/bash
CURRENT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))


DEVICE?=GPU
MODEL_SIZE?=m
MODEL_NAME=yolov8${MODEL_SIZE}
DATA_TYPE?=FP16
INPUT?=/dev/video0
#----------------------------------------------------------------------------------------------------------------------
# Docker Settings
#----------------------------------------------------------------------------------------------------------------------
DOCKER_IMAGE_NAME=telefonica_yolov8_evaluation
export DOCKER_BUILDKIT=1

DOCKER_RUN_PARAMS= \
	-it --rm -a stdout -a stderr -e DISPLAY=${DISPLAY}  \
	--privileged -v /dev:/dev \
	-v ${CURRENT_DIR}/models:/workspace/models \
	-v ${CURRENT_DIR}/videos:/workspace/videos \
	-v /tmp/.X11-unix:/tmp/.X11-unix  -v ${HOME}/.Xauthority:/home/root/.Xauthority \
	${DOCKER_IMAGE_NAME}
	
#----------------------------------------------------------------------------------------------------------------------
# Targets
#----------------------------------------------------------------------------------------------------------------------
default: benchmark
.PHONY:  

build: 
	@$(call msg, Building Docker image ${DOCKER_IMAGE_NAME} ...)
	@docker build --rm . -t ${DOCKER_IMAGE_NAME}
	
fetch:
	@$(call msg, Downloading the ${MODEL_NAME} model ...)
	@docker run ${DOCKER_RUN_PARAMS} \
		wget -nc -q -P /workspace/models/${MODEL_NAME} \
			https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8${MODEL_SIZE}.pt

ir: fetch
	@$(call msg, Converting the ${MODEL_NAME} model to IR ...)
	@docker run ${DOCKER_RUN_PARAMS} bash -c '\
		python3 ./utils/mo.py --model=./models/${MODEL_NAME}/${MODEL_NAME}.pt --data_type="${DATA_TYPE}" && \
		 mv ./models/${MODEL_NAME}/${MODEL_NAME}_openvino_model ./models/${MODEL_NAME}/${DATA_TYPE}'
	
benchmark: ir
	@$(call msg, Benchmarking the ${MODEL_NAME} model ...)
	@docker run ${DOCKER_RUN_PARAMS} bash -c '\
		benchmark_app \
				-m ./models/${MODEL_NAME}/${DATA_TYPE}/${MODEL_NAME}.xml \
				-shape [1,3,640,640] \
				-d ${DEVICE} \
				-t 10 '

demo: build fetch
	@$(call msg, Running the yolov8 object detection demo ...)
	@xhost +
	@docker run ${DOCKER_RUN_PARAMS} \
		python3 ./main.py  \
				--model ./models/${MODEL_NAME}/${MODEL_NAME}.pt \
				--device ${DEVICE} \
				--input ${INPUT}
#----------------------------------------------------------------------------------------------------------------------
# helper functions
#----------------------------------------------------------------------------------------------------------------------
define msg
	tput setaf 2 && \
	for i in $(shell seq 1 120 ); do echo -n "-"; done; echo  "" && \
	echo "         "$1 && \
	for i in $(shell seq 1 120 ); do echo -n "-"; done; echo "" && \
	tput sgr0
endef

