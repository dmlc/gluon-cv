Installation
------------

Install MXNet
^^^^^^^^^^^^^

.. Ignore prerequisites to make the index page concise, which will be shown at
   the install page

.. raw:: html

   <style>.admonition-prerequisite {display: none;}</style>

.. include:: /install/install-include.rst

Check :doc:`/install/install-more` for more installation instructions and options.

Install GluonCV
^^^^^^^^^^^^^^^^

The easiest way to install GluonCV is through `pip <https://pip.pypa.io/en/stable/installing/>`_.

.. code-block:: bash

 pip install gluoncv --upgrade

 # if you are eager to try new features, try nightly build instead

 pip install gluoncv --pre --upgrade

.. hint::

  Nightly build is updated daily around 12am UTC to match master progress.

  Optionally, you can clone the GluonCV project and install it locally

  .. code-block:: bash

    git clone https://github.com/dmlc/gluon-cv
    cd gluon-cv && python setup.py install --user
