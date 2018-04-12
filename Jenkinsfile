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
      python setup.py install
      cd docs && make html
      aws s3 sync --delete build/html/ s3://gluon-vision.mxnet.io/ --acl public-read
      """
    }
  }
}
}
