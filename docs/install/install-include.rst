.. role:: title
.. role:: opt
   :class: option
.. role:: act
   :class: active option

.. container:: install

   .. container:: opt-group

      :title:`Platform:`
      :act:`Local`
      :opt:`Cloud`
      :opt:`Devices`

   .. container:: devices

      .. container:: opt-group

         :title:`Device:`
         :act:`Raspberry Pi`
         :opt:`Jetson`

         .. container:: raspberry-pi

            .. card::
               :title: Installation Guide
               :link: /install/install-pi.html

               How to install MXNet on a Raspberry Pi 3.

         .. container:: jetson

            .. card::
               :title: Installation Guide
               :link: /install/install-jetson.html

               How to install MXNet on a Jetson TX.

   .. container:: cloud

      .. container:: opt-group

         :title:`Provider:`
         :act:`Alibaba`
         :opt:`AWS`
         :opt:`Google Cloud`
         :opt:`Microsoft Azure`
         :opt:`Oracle Cloud`

      .. admonition:: Installation Guides:

         .. container:: alibaba

               - `NVIDIA VM for Alibaba <https://docs.nvidia.com/ngc/ngc-alibaba-setup-guide/launching-nv-cloud-vm-console.html#launching-nv-cloud-vm-console>`_

         .. container:: aws

               - `AWS Deep Learning AMI
                 <https://aws.amazon.com/machine-learning/amis/>`_: preinstalled Conda environments for Python 2 or 3 with MXNet, CUDA, cuDNN, MKL-DNN, and AWS Elastic Inference
               - `Amazon SageMaker <https://aws.amazon.com/sagemaker/>`_: managed training and deployment of MXNet models
               - `Dynamic Training on AWS <https://github.com/awslabs/dynamic-training-with-apache-mxnet-on-aws>`_: experimental manual EC2 setup or semi-automated CloudFormation setup
               - `NVIDIA VM for AWS <https://aws.amazon.com/marketplace/pp/B076K31M1S>`_

         .. container:: google-cloud

               - `NVIDIA VM for Google Cloud <https://console.cloud.google.com/marketplace/details/nvidia-ngc-public/nvidia_gpu_cloud_image>`_

         .. container:: microsoft-azure

               - `NVIDIA VM for Azure <https://azuremarketplace.microsoft.com/en-us/marketplace/apps/nvidia.ngc_azure_17_11?tab=Overview>`_

         .. container:: oracle-cloud

               - `NVIDIA VM for Oracle Cloud <https://docs.cloud.oracle.com/iaas/Content/Compute/References/ngcimage.htm>`_

   .. container:: local

      .. container:: opt-group

         :title:`OS:`
         :opt:`Linux`
         :opt:`macOS`
         :opt:`Windows`

      .. container:: opt-group

         :title:`Package:`
         :act:`Pip`
         :opt:`Docker`


      .. container:: opt-group

         :title:`Backend:`
         :act:`Native`
         :opt:`CUDA`
         :opt:`MKL-DNN`
         :opt:`CUDA + MKL-DNN`

         .. raw:: html

            <div class="mdl-tooltip" data-mdl-for="native">Build-in backend for CPU.</div>
            <div class="mdl-tooltip" data-mdl-for="cuda">Required to run on Nvidia GPUs.</div>
            <div class="mdl-tooltip" data-mdl-for="mkl-dnn">Accelerate Intel CPU performance.</div>
            <div class="mdl-tooltip" data-mdl-for="cuda-mkl-dnn">Enable both Nvidia CPUs and Inter CPU acceleration.</div>

      .. admonition:: Prerequisites:

         .. container:: docker

            - Requires `docker <https://docs.docker.com/install/>`_
              and Docker can be used by a non-root user.

         .. container:: docker

              .. container:: cuda cuda-mkl-dnn

                 - `nvidia-docker
                   <https://github.com/NVIDIA/nvidia-docker>`_ is required to
                   run on Nvidia GPUs.

         .. container:: pip

            - Requires `pip >= 9. <https://pip.pypa.io/en/stable/installing/>`_.
              Both Python 2 and Python 3 are supported.
            - Hint: append the flag ``--pre`` at the end of the command will
              install the nightly build.
            .. - Hint: refer to `Issue 8671
               <https://github.com/apache/incubator-mxnet/issues/8671>`_ for
               all MXNet variants that available for pip.

            .. container:: cuda cuda-mkl-dnn

               - Requires `CUDA
                 <https://developer.nvidia.com/cuda-toolkit-archive>`_.
                 Supported versions include 8.0, 9.0, and 9.2.
               - Hint: `cuDNN <https://developer.nvidia.com/cudnn>`_ is already
                 included in the MXNet binary, so you don't need to install it.

            .. container:: mkl-dnn cuda-mkl-dnn

               - Hint: `MKL-DNN <https://01.org/mkl-dnn>`_ is already included in
                 the MXNet binary, so you don't need to install it.
               - For detailed information on MKL and MKL-DNN,
                 refer to the `MKLDNN_README <https://mxnet.incubator.apache.org/versions/master/tutorials/mkldnn/MKLDNN_README.html>`_.

      .. admonition:: Command:

         .. container:: pip

            .. container:: native

               .. code-block:: bash

                  pip install mxnet

            .. container:: cuda

               .. code-block:: bash

                  # Here we assume CUDA 9.2 is installed. You can change the number
                  # according to your own CUDA version.
                  pip install mxnet-cu92

            .. container:: mkl-dnn

               .. code-block:: bash

                  pip install mxnet-mkl

            .. container:: cuda-mkl-dnn

               .. code-block:: bash

                  # Here we assume CUDA 9.2 is installed. You can change the number
                  # according to your own CUDA version.
                  pip install mxnet-cu92mkl

         .. container:: docker

            .. container:: native

               .. code-block:: bash

                  docker pull mxnet/python

            .. container:: cuda

               .. code-block:: bash

                  docker pull mxnet/python:gpu

            .. container:: mkl-dnn

               .. code-block:: bash

                  docker pull mxnet/python:1.3.0_cpu_mkl

            .. container:: cuda-mkl-dnn

               .. code-block:: bash

                   docker pull mxnet/python:1.3.0_gpu_cu90_mkl_py3

.. raw:: html

   <style>.disabled { display: none; }</style>
   <script type="text/javascript" src='../_static/install-options.js'></script>
