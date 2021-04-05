#!/bin/bash

TYPE=$1

if [ -z $TYPE ]; then
	echo "No type detected. Choices: cpu, gpu"
	exit 1
fi;

if [ $TYPE == cpu ] || [ $TYPE == CPU ]; then
	docker build --no-cache -f Dockerfile.cpu -t gluon-cv-1:cpu-latest .
	docker tag gluon-cv-1:cpu-latest $AWS_ECR_REPO:cpu-latest
	docker push $AWS_ECR_REPO:cpu-latest
elif [ $TYPE == gpu ] || [ $TYPE == GPU ]; then
	docker build --no-cache -f Dockerfile.gpu -t gluon-cv-1:latest .
	docker tag gluon-cv-1:latest $AWS_ECR_REPO:latest
	docker push $AWS_ECR_REPO:latest
else
	echo "Invalid type detected. Choices: cpu, gpu"
	exit 1
fi;
