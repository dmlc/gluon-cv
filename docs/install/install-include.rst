Select your preferences and run the install command.

.. role:: title
.. role:: opt
   :class: option
.. role:: act
   :class: active option

.. container:: install

  .. container:: opt-group

     :title:`OS:`
     :opt:`Linux`
     :opt:`macOS`
     :opt:`Windows`

  .. container:: opt-group

     :title:`Version:`
     :act:`Stable`
     :opt:`Nightly`
     :opt:`Source`

     .. raw:: html

        <div class="mdl-tooltip" data-mdl-for="stable">Stable Release.</div>
        <div class="mdl-tooltip" data-mdl-for="nightly">Nightly build with latest features.</div>
        <div class="mdl-tooltip" data-mdl-for="source">Install GluonCV from source.</div>


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
        <div class="mdl-tooltip" data-mdl-for="cuda-mkl-dnn">Enable both Nvidia GPUs and Intel CPU acceleration.</div>

  .. admonition:: Prerequisites:

     - Requires `pip >= 9. <https://pip.pypa.io/en/stable/installing/>`_.

     - Note that you can install the extra optional requirements all together by replacing "pip install gluoncv" with "pip install gluoncv[full]".

     .. container:: nightly

        - Nightly build provides latest features for enthusiasts.

  .. admonition:: Command:

     .. container:: stable

        .. container:: native

           .. code-block:: bash

              # for mxnet
              pip install --upgrade mxnet
              # for pytorch
              pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

              pip install --upgrade gluoncv

        .. container:: cuda

           .. code-block:: bash

              # Here we assume CUDA 10.2 is installed. You can change the number
              # according to your own CUDA version.

              # for mxnet
              pip install --upgrade mxnet-cu102
              # for pytorch
              pip install torch==1.6.0 torchvision==0.7.0

              pip install --upgrade gluoncv

        .. container:: mkl-dnn

           .. code-block:: bash

              pip install --upgrade mxnet-mkl gluoncv

        .. container:: cuda-mkl-dnn

           .. code-block:: bash

              # Here we assume CUDA 10.2 is installed. You can change the number
              # according to your own CUDA version.
              pip install --upgrade mxnet-cu102mkl gluoncv

     .. container:: nightly

        .. container:: native

           .. code-block:: bash

              # for mxnet
              pip install --upgrade mxnet -f https://dist.mxnet.io/python/all
              # for pytorch
              pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

              pip install --pre --upgrade gluoncv

        .. container:: cuda

           .. code-block:: bash

              # mxnet
              pip install --upgrade mxnet-cu102 -f https://dist.mxnet.io/python/all
              # pytorch
              pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html

              pip install --pre --upgrade gluoncv

        .. container:: mkl-dnn

           .. code-block:: bash

              pip install --pre --upgrade gluoncv
              pip install --upgrade mxnet-mkl -f https://dist.mxnet.io/python/all

        .. container:: cuda-mkl-dnn

           .. code-block:: bash

               pip install --pre --upgrade gluoncv
               pip install --upgrade mxnet-cu102mkl -f https://dist.mxnet.io/python/all

     .. container:: source

        .. container:: native

           .. code-block:: bash

              # mxnet
              pip install --upgrade mxnet -f https://dist.mxnet.io/python/all
              # pytorch
              pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

              git clone https://github.com/dmlc/gluon-cv
              cd gluon-cv && python setup.py install --user

        .. container:: cuda

           .. code-block:: bash

              # mxnet
              pip install --upgrade mxnet-cu100 -f https://dist.mxnet.io/python/all
              # pytorch
              pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html

              git clone https://github.com/dmlc/gluon-cv
              cd gluon-cv && python setup.py install --user

        .. container:: mkl-dnn

           .. code-block:: bash

              pip install --upgrade mxnet-mkl -f https://dist.mxnet.io/python/all
              git clone https://github.com/dmlc/gluon-cv
              cd gluon-cv && python setup.py install --user

        .. container:: cuda-mkl-dnn

           .. code-block:: bash

               pip install --upgrade mxnet-cu102mkl -f https://dist.mxnet.io/python/all
               git clone https://github.com/dmlc/gluon-cv
               cd gluon-cv && python setup.py install --user
