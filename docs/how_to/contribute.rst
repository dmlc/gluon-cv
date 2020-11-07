Contribute to GluonCV
=====================


GluonCV community is more than welcome to receive contributions from everyone!
Latest documentation can be found in `GluonCV website <http://gluon-cv.mxnet.io/index.html>`_.

There are lots of opportunities for you to become our
`contributors <https://github.com/dmlc/gluon-cv/graphs/contributors>`_:

-   Ask or answer questions on `GitHub
    issues <https://github.com/dmlc/gluon-cv/issues>`_.
-   Propose ideas, or review proposed design ideas on `GitHub
    issues <https://github.com/dmlc/gluon-cv/issues>`_.
-   Improve the
    `documentation <http://gluon-cv.mxnet.io/index.html>`_.
-   Contribute bug reports `GitHub
    issues <https://github.com/dmlc/gluon-cv/issues>`_.
-   Write new
    `scripts <https://github.com/dmlc/gluon-cv/tree/master/scripts>`_ to
    reproduce state-of-the-art results.
-   Write new
    `tutorials <https://github.com/dmlc/gluon-cv/tree/master/docs/tutorials>`_.
-   Write new `public
    datasets <https://github.com/dmlc/gluon-cv/tree/master/gluoncv/data>`_.
-   Submitting new computer vision `algorithm <https://github.com/dmlc/gluon-cv/tree/master/gluoncv/model_zoo>`_.
-   Most importantly, if you have an idea of how to contribute, then do
    it!

For a list of open starter tasks, check `good first
issues <https://github.com/dmlc/gluon-cv/labels/good%20first%20issue>`_.

How-to
------

-   `Make Changes`_
-   `Contribute Scripts`_
-   `Contribute Tutorials`_
-   `Contribute Docs`_
-   `Contribute new CV algorithms`_
-   `Contribute new API`_
-   `Git Workflow Howtos`_

    -   `How to submit pull request`_
    -   `How to resolve conflict with master`_
    -   `How to combine multiple commits into one`_
    -   `What is the consequence of force push`_


.. `Make Changes`_:

Make changes
------------

Our package uses continuous integration and code coverage tools for
verifying pull requests. Before submitting, contributor should perform
the following checks:

-   `Lint (code style)
    check <https://github.com/dmlc/gluon-cv/blob/master/Jenkinsfile#L6-L11>`_.
-   `Py2 <https://github.com/dmlc/gluon-cv/blob/master/Jenkinsfile#L23-L35>`_
    and
    `Py3 <https://github.com/dmlc/gluon-cv/blob/master/Jenkinsfile#L54-L66>`_ tests.

.. `Contribute Scripts`_:

Contribute Scripts
------------------

The `scripts <https://github.com/dmlc/gluon-cv/tree/master/scripts>`_ in
GluonCV are typically for reproducing state-of-the-art (SOTA) results,
or for a simple and interesting application. They are intended for
practitioners who are familiar with the libraries to tweak and hack. For
SOTA scripts, we usually request training scripts to be uploaded
`here <https://github.com/dmlc/web-data/tree/master/gluoncv/logs>`_, and
then linked to in the example documentation.

See `existing examples <https://github.com/dmlc/gluon-cv/tree/master/scripts>`_.


.. `Contribute Tutorials`_:

Contribute Tutorials
--------------------

Our `website tutorials <https://gluon-cv.mxnet.io/build/examples_classification/index.html>`_ are
intended for people who are interested in CV and want to get better
familiarized on different parts in CV. In order for people to easily
understand the content, the code needs to be clean and readable,
accompanied by good quality writing.

See `existing
examples available <https://gluon-cv.mxnet.io/build/examples_classification/index.html>`_.

.. `Contribute new CV algorithms`_:

Contribute new API
------------------

There are several different types of APIs, such as \*model definition
APIs, public dataset APIs, and building block APIs\*.

*Model definition APIs* facilitate the sharing of pre-trained models. If
you\'d like to contribute models with pre-trained weights, you can `open
an issue <https://github.com/dmlc/gluon-cv/issues/new>`_ and ping
committers first, we will help with things such as hosting the model
weights while you propose the patch.

*Public dataset APIs* facilitate the sharing of public datasets. Like
model definition APIs, if you\'d like to contribute new public datasets,
you can `open an issue <https://github.com/dmlc/gluon-cv/issues/new>`_
and ping committers and review the dataset needs. If you\'re unsure,
feel free to open an issue anyway.

Finally, our *data and model building block APIs* come from repeated
patterns in examples. It has the highest quality bar and should always
starts from a good design. If you have an idea on proposing a new API,
we encourage you to `draft a design proposal
first <https://github.com/dmlc/gluon-cv/labels/enhancement>`_, so that
the community can help iterate. Once the design is finalized, everyone
who are interested in making it happen can help by submitting patches.
For designs that require larger scopes, we can help set up GitHub
project to make it easier for others to join.


.. `Contribute Docs`_:

Contribute Docs
---------------

Documentation is no less important than code. Good documentation
delivers the correct message clearly and concisely. If you see any issue
in the existing documentation, a patch to fix is most welcome! To locate
the code responsible for the doc, you may use \"View page source\" in
the top right corner, or the \"\[source\]\" links after each API. Also,
\"\[git grep\]\" works nicely if there\'s unique string.


.. `Contribute new CV algorithms`_:

Contribute new CV algorithms
----------------------------

Officially supported algorithms in GluonCV consist of the following five components:

- `Model definition <https://github.com/dmlc/gluon-cv/tree/estimator/gluoncv/model_zoo>`_. Models are written in `mxnet.gluon.HybridBlock` or `mxnet.gluon.Block`. You may get a better idea to start with the implementation of famous `resnet <https://github.com/dmlc/gluon-cv/blob/estimator/gluoncv/model_zoo/resnetv1b.py`_ models. In addition to base `ResNetV1b` class definition, we also encourage the definition of individual network with names so that consumption of specific model variant is made easy. For example: `def resnet18_v1b <https://github.com/dmlc/gluon-cv/blob/estimator/gluoncv/model_zoo/resnetv1b.py#L268>`_
- Accompanying datasets, preprocessing function, data augmentation, evaluation metric, loss functions. You will only need to implement them in case you don't find appropriate ones in the existing library.
    - `dataset definitions <https://github.com/dmlc/gluon-cv/tree/estimator/gluoncv/data>`_
    - `data transformations <https://github.com/dmlc/gluon-cv/tree/estimator/gluoncv/data/transforms>`_
    - `data transformation presets for specific algorithms <https://github.com/dmlc/gluon-cv/tree/estimator/gluoncv/data/transforms/presets>`_
    - `reusable neural network components <https://github.com/dmlc/gluon-cv/tree/estimator/gluoncv/nn>`_
    - `utils <https://github.com/dmlc/gluon-cv/tree/estimator/gluoncv/utils>`_: anything else you find it's not appropriate to reside in other major components
- `Python Scripts <https://github.com/dmlc/gluon-cv/tree/estimator/scripts>`_. Scripts folder includes training/evaluation/demo python scripts that people can play with and modify. You can use these script to train/evaluate models in development as well.
- `Pretrained weights <https://github.com/dmlc/gluon-cv/blob/estimator/gluoncv/model_zoo/model_store.py>`_: organized by hashing codes, the pretrained weights are hosted on S3 bucket for all users to consume the models directly. Obviously, do not discard the best `params` file you have trained and evaluated, share with us in the PR, the organizers can help upload the weights and make it new algorithm easier to digest!
- `Documents <https://github.com/dmlc/gluon-cv/tree/estimator/docs>`_. Let more people to know your new shining algorithm by adding the scores to the tables in model zoo website. `Contribute Tutorials`_ for inference/training can definitely flatten the learning curve of users to adopt the new models.

About hybrid/non-hybrid models in GluonCV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GluonCV recommends the hybrid version of network(`context of hybrid network <https://mxnet.apache.org/api/python/docs/tutorials/packages/gluon/blocks/hybridize.html>`_) for GluonCV built-in algorithm. The advantage of fully hybrid network is to enable hassle-free deployment outside of python ecosystem. The major limitation of fully hybrid network is the restricted usage of accessing shape of tensors inside the network. `shape_array <https://mxnet.apache.org/api/python/docs/api/ndarray/ndarray.html?highlight=shape_array#mxnet.ndarray.shape_array>`_ operator is used in many cases though it's not fully interchangeable with a normal `.shape` attribute. Don't worry, we are address this issue and feel free to discuss with existing contributors and committers for situations where fully hybrid network isn't applicable.


.. `Git Workflow Howtos`_:

Git Workflow Howtos
-------------------

.. `How to submit pull request`_:

How to submit pull request
~~~~~~~~~~~~~~~~~~~~~~~~~~

-   Before submit, please rebase your code on the most recent version of
    master, you can do it by

    .. code-block:: bash

      git remote add upstream https://github.com/dmlc/gluon-cv
      git fetch upstream
      git rebase upstream/master


-   If you have multiple small commits, it might be good to merge them
    together(use git rebase then squash) into more meaningful groups.
-   Send the pull request!
    -   Fix the problems reported by automatic checks
    -   If you are contributing a new module or new function, add a test.

.. `How to resolve conflict with master`_:

How to resolve conflict with master
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-   First rebase to most recent master

    .. code-block:: bash

        # The first two steps can be skipped after you do it once.
        git remote add upstream https://github.com/dmlc/gluon-cv
        git fetch upstream
        git rebase upstream/master


-   The git may show some conflicts it cannot merge, say
    `conflicted.py`.

    -   Manually modify the file to resolve the conflict.
    -   After you resolved the conflict, mark it as resolved by

    .. code-block:: bash

        git add conflicted.py


-   Then you can continue rebase by

    .. code-block:: bash

      git rebase --continue

-   Finally push to your fork, you may need to force push here.

    .. code-block:: bash

        git push --force


.. `How to combine multiple commits into one`_:

How to combine multiple commits into one
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes we want to combine multiple commits, especially when later
commits are only fixes to previous ones, to create a PR with set of
meaningful commits. You can do it by following steps. - Before doing so,
configure the default editor of git if you haven't done so before.

.. code-block:: bash

    git config core.editor the-editor-you-like


-   Assume we want to merge last 3 commits, type the following commands

    .. code-block:: bash

        git rebase -i HEAD~3


-   It will pop up an text editor. Set the first commit as `pick`, and
    change later ones to `squash`.
-   After you saved the file, it will pop up another text editor to ask
    you modify the combined commit message.
-   Push the changes to your fork, you need to force push.

    .. code-block:: bash

        git push --force


Reset to the most recent master
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can always use git reset to reset your version to the most recent
master. Note that all your **\*local changes will get lost**\*. So only
do it when you do not have local changes or when your pull request just
get merged.

.. code-block:: bash

    git reset --hard [hash tag of master]
    git push --force


.. `What is the consequence of force push`_:

What is the consequence of force push
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The previous two tips requires force push, this is because we altered
the path of the commits. It is fine to force push to your own fork, as
long as the commits changed are only yours.
