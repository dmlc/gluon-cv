# Updating the Docker Image for AWS Batch

Our current batch job dockers are in 985964311364.dkr.ecr.us-east-1.amazonaws.com/gluon-cv-1. To update the docker:

- update the Dockerfile
- Make sure docker and docker-compose, as well as the docker python package are installed.
- Export the AWS account credentials as environment variables
- CD to the same folder as the Dockerfile and execute the following:

```shell
# This executes a command that logs into ECR.
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 985964311364.dkr.ecr.us-east-1.amazonaws.com

# Following script will build, tag, and push the image
# For cpu
./docker_deploy.sh cpu
# For gpu
./docker_deploy.sh gpu

```

