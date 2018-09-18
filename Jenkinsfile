stage("Sanity Check") {
  node {
    ws('workspace/gluon-cv-lint') {
      checkout scm
      sh """#!/bin/bash
      set -e
      conda env update -n gluon_vision_pylint -f tests/pylint.yml
      conda activate gluon_vision_pylint
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
        set -e
        # conda env remove -n gluon_cv_py2_test -y
        # conda env create -n gluon_cv_py2_test -f tests/py2.yml
        conda env update -n gluon_cv_py2_test -f tests/py2.yml
        conda activate gluon_cv_py2_test
        conda list
        make clean
        # from https://stackoverflow.com/questions/19548957/can-i-force-pip-to-reinstall-the-current-version
        pip install --upgrade --force-reinstall --no-deps .
        env
        export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64
        export MPLBACKEND=Agg
        nosetests --with-coverage --cover-package gluoncv -v tests/unittests
        """
      }
    }
  },
  'Python 3': {
    node {
      ws('workspace/gluon-cv-py3') {
        checkout scm
        sh """#!/bin/bash
        set -e
        # conda env remove -n gluon_cv_py3_test -y
        # conda env create -n gluon_cv_py3_test -f tests/py3.yml
        conda env update -n gluon_cv_py3_test -f tests/py3.yml
        conda activate gluon_cv_py3_test
        conda list
        make clean
        # from https://stackoverflow.com/questions/19548957/can-i-force-pip-to-reinstall-the-current-version
        pip install --upgrade --force-reinstall --no-deps .
        env
        export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64
        export MPLBACKEND=Agg
        nosetests --with-coverage --cover-package gluoncv -v tests/unittests
        rm -f coverage.svg
        coverage-badge -o coverage.svg
        if [[ ${env.BRANCH_NAME} == master ]]; then
            aws s3 cp coverage.svg s3://gluon-cv.mxnet.io/coverage.svg --acl public-read --cache-control no-cache
            echo "Uploaded coverage badge to http://gluon-cv.mxnet.io"
        else
            aws s3 cp coverage.svg s3://gluon-vision-staging/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/coverage.svg --acl public-read --cache-control no-cache
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
      conda env update -n gluon_vision_docs -f docs/build.yml
      conda activate gluon_vision_docs
      export PYTHONPATH=\${PWD}
      env
      export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64
      git clean -fx
      cd docs && make clean && make html

      if [[ ${env.BRANCH_NAME} == master ]]; then
          aws s3 cp s3://gluon-cv.mxnet.io/coverage.svg build/html/coverage.svg
          aws s3 sync --delete build/html/ s3://gluon-cv.mxnet.io/ --acl public-read
          aws s3 cp build/html/coverage.svg s3://gluon-cv.mxnet.io/coverage.svg --acl public-read --cache-control no-cache
          echo "Uploaded doc to http://gluon-cv.mxnet.io"
      else
          aws s3 cp s3://gluon-vision-staging/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/coverage.svg build/html/coverage.svg
          aws s3 sync --delete build/html/ s3://gluon-vision-staging/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/ --acl public-read
          echo "Uploaded doc to http://gluon-vision-staging.s3-website-us-west-2.amazonaws.com/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/index.html"
      fi
      """

      if (env.BRANCH_NAME.startsWith("PR-")) {
        pullRequest.comment("Job ${env.BRANCH_NAME}-${env.BUILD_NUMBER} is done. \nDocs are uploaded to http://gluon-vision-staging.s3-website-us-west-2.amazonaws.com/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/index.html \nCode coverage of this PR: ![pr.svg](http://gluon-vision-staging.s3-website-us-west-2.amazonaws.com/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/coverage.svg?) vs. Master: ![master.svg](http://gluon-cv.mxnet.io/coverage.svg?)")
      }
    }
  }
}
