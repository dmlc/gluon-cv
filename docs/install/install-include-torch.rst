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

     .. raw:: html

        <div class="mdl-tooltip" data-mdl-for="native">Build-in backend for CPU.</div>
        <div class="mdl-tooltip" data-mdl-for="cuda">Required to run on Nvidia GPUs.</div>

  .. admonition:: Prerequisites:

     - Requires `pip >= 9. <https://pip.pypa.io/en/stable/installing/>`_.
       Both Python 2 and Python 3 are supported.

     .. container:: nightly

        - Nightly build provides latest features for enthusiasts.

  .. admonition:: Command:

     .. container:: stable

        .. container:: native

           .. code-block:: bash

              pip install --upgrade gluoncv
              pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

        .. container:: cuda

           .. code-block:: bash

              # Here we assume CUDA 10.2 is installed. You can change the number
              # according to your own CUDA version.
              pip install --upgrade gluoncv
              pip install torch==1.6.0 torchvision==0.7.0


     .. container:: nightly

        .. container:: native

           .. code-block:: bash

              pip install --pre --upgrade gluoncv
              pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

        .. container:: cuda

           .. code-block:: bash

              pip install --pre --upgrade gluoncv
              pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html


     .. container:: source

        .. container:: native

           .. code-block:: bash

              pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
              git clone https://github.com/dmlc/gluon-cv
              cd gluon-cv && python setup.py install --user

        .. container:: cuda

           .. code-block:: bash

              pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html
              git clone https://github.com/dmlc/gluon-cv
              cd gluon-cv && python setup.py install --user
