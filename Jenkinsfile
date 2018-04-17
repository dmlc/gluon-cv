stage("LINT") {
  node {
    ws('workspace/gluon-vision-lint') {
      checkout scm
      sh """#!/bin/bash
      set -e
      conda env update -f tests/pylint.yml
      source activate gluon_vision_pylint
      conda list
      make pylint
      """
    }
  }
}


if (env.BRANCH_NAME == "master") {
stage("Docs") {
  node {
    ws('workspace/gluon-vision-docs') {
      checkout scm
      sh """#!/bin/bash
      set -e
      conda env update -f docs/build.yml
      source activate gluon_vision_docs
      pip uninstall gluonvision
      python setup.py install
      env
      export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64
      cd docs && make html
      aws s3 sync --delete build/html/ s3://gluon-vision.mxnet.io/ --acl public-read
      """
    }
  }
}
}
