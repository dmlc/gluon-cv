#!/bin/bash

TYPE=$1

if [ -z $TYPE ]; then
	echo "No type detected. Choices: cpu, gpu"
	exit 1
fi;

if [ $TYPE == cpu ] || [ $TYPE == CPU ]; then
	docker build -f Dockerfile.cpu -t gluon-cv-1:cpu-latest .
	docker tag gluon-cv-1:cpu-latest 985964311364.dkr.ecr.us-east-1.amazonaws.com/gluon-cv-1:cpu-latest
	docker push 985964311364.dkr.ecr.us-east-1.amazonaws.com/gluon-cv-1:cpu-latest
elif [ $TYPE == gpu ] || [ $TYPE == GPU ]; then
	docker build -f Dockerfile.gpu -t gluon-cv-1:latest .
	docker tag gluon-cv-1:latest 985964311364.dkr.ecr.us-east-1.amazonaws.com/gluon-cv-1:latest
	docker push 985964311364.dkr.ecr.us-east-1.amazonaws.com/gluon-cv-1:latest
else
	echo "Invalid type detected. Choices: cpu, gpu"
	exit 1
fi;
