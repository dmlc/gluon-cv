stage("Sanity Check") {
  node {
    ws('workspace/gluon-cv-lint') {
      checkout scm
      sh """#!/bin/bash
      set -e
      conda env update -f tests/pylint.yml
      source activate gluon_vision_pylint
      conda list
      make clean
      make pylint
      """
    }
  }
}

stage("Unit Test") {
  parallel 'Python 2': {
    node {
      ws('workspace/gluon-cv-py2') {
        checkout scm
        sh """#!/bin/bash
        conda env update -f tests/py2.yml
        source activate gluon_cv_py2
        conda list
        make clean
        python setup.py install
        nosetests --with-coverage --cover-package gluoncv -v tests/unittests
        """
      }
    }
  }
  parallel 'Python 3': {
    node {
      ws('workspace/gluon-cv-py3') {
        checkout scm
        sh """#!/bin/bash
        conda env update -f tests/py3.yml
        source activate gluon_cv_py3
        conda list
        make clean
        python setup.py install
        nosetests --with-coverage --cover-package gluoncv -v tests/unittests
        coverage-badge -o coverage.svg
        if [[ ${env.BRANCH_NAME} == master ]]; then
            aws s3 cp coverage.svg s3://gluon-cv.mxnet.io/coverage.svg --acl public-read
            echo "Uploaded coverage badge to http://gluon-cv.mxnet.io"
        else
            aws s3 cp coverage.svg s3://gluon-vision-staging/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/coverage.svg --acl public-read
            echo "Uploaded coverage badge to http://gluon-vision-staging.s3-website-us-west-2.amazonaws.com/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/coverage.svg"
        fi
        """
      }
    }
  }
}


stage("Build Docs") {
  node {
    ws('workspace/gluon-cv-docs') {
      checkout scm
      sh """#!/bin/bash
      set -e
      set -x
      conda env update -f docs/build.yml
      source activate gluon_vision_docs
      export PYTHONPATH=\${PWD}
      env
      export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64
      cd docs && make clean && make html

      if [[ ${env.BRANCH_NAME} == master ]]; then
          aws s3 sync --delete build/html/ s3://gluon-cv.mxnet.io/ --acl public-read
          echo "Uploaded doc to http://gluon-cv.mxnet.io"
      else
          aws s3 sync --delete build/html/ s3://gluon-vision-staging/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/ --acl public-read
          echo "Uploaded doc to http://gluon-vision-staging.s3-website-us-west-2.amazonaws.com/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/index.html"
      fi
      """

      if (env.BRANCH_NAME.startsWith("PR-")) {
        pullRequest.comment("Job ${env.BRANCH_NAME}-${env.BUILD_NUMBER} is done. \nDocs are uploaded to http://gluon-vision-staging.s3-website-us-west-2.amazonaws.com/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/index.html \nCode coverage of this PR: [pr.svg](http://gluon-vision-staging.s3-website-us-west-2.amazonaws.com/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/coverage.svg) vs. Master: [master.svg](s3://gluon-cv.mxnet.io/coverage.svg)")
      }
    }
  }
}
