# Docker Support in GluonCV

We provide the [Docker](https://www.docker.com/) container with everything set up to run GluonCV. With the prebuilt docker image, there is no need to worry about the operating systems or system dependencies. You can launch a [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) development environment and try out to use GluonCV to solve your problem.

## Run Docker

You can run the docker with the following command.

```
docker pull gluonai/gluon-cv:gpu-latest
docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 --shm-size=2g gluonai/gluon-cv:gpu-latest
```

Here, we open the ports 8888, 8787, 8786, which are used for connecting to JupyterLab. Also, we set `--shm-size` to `2g`. This sets the shared memory storage to 2GB. Since NCCL will create shared memory segments, this argument is essential for the JupyterNotebook to work with NCCL. (See also https://github.com/NVIDIA/nccl/issues/290).

## Build your own Docker Image

To build a docker image fom the dockerfile, you may use the following command:

```
docker build -t gluonai/gluon-cv:gpu-latest .
```