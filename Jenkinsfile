max_time = 120
stage("Sanity Check") {
  node {
    ws('workspace/gluon-cv-lint') {
      timeout(time: max_time, unit: 'MINUTES') {
        checkout scm
        sh """#!/bin/bash
        set -e
        conda env update -n gluon_vision_pylint -f tests/pylint.yml --prune
        conda activate gluon_vision_pylint
        conda list
        make clean
        make pylint
        """
      }
    }
  }
}

stage("Unit Test") {
  parallel 'Python 2': {
    node('linux-gpu') {
      ws('workspace/gluon-cv-py2') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
          # old pip packages won't be cleaned: https://github.com/conda/conda/issues/5887
          # remove and create new env instead
          set -ex
          conda env remove -n gluon_cv_py2_test
          conda env create -n gluon_cv_py2_test -f tests/py2.yml
          conda env update -n gluon_cv_py2_test -f tests/py2.yml --prune
          conda activate gluon_cv_py2_test
          conda list
          export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
          make clean
          # from https://stackoverflow.com/questions/19548957/can-i-force-pip-to-reinstall-the-current-version
          pip install --upgrade --force-reinstall --no-deps .
          env
          export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64
          export MPLBACKEND=Agg
          export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
          nosetests --with-timer --timer-ok 5 --timer-warning 20 -x --with-coverage --cover-package gluoncv -v tests/unittests
          """
        }
      }
    }
  },
  'Python 3': {
    node('linux-gpu') {
      ws('workspace/gluon-cv-py3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
          set -ex
          # remove and create new env instead
          conda env remove -n gluon_cv_py3_test
          conda env create -n gluon_cv_py3_test -f tests/py3.yml
          conda env update -n gluon_cv_py3_test -f tests/py3.yml --prune
          conda activate gluon_cv_py3_test
          conda list
          export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
          make clean
          # from https://stackoverflow.com/questions/19548957/can-i-force-pip-to-reinstall-the-current-version
          pip install --upgrade --force-reinstall --no-deps .
          env
          export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64
          export MPLBACKEND=Agg
          export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
          nosetests --with-timer --timer-ok 5 --timer-warning 20 -x --with-coverage --cover-package gluoncv -v tests/unittests
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
}


stage("Build Docs") {
  node('linux-gpu') {
    ws('workspace/gluon-cv-docs') {
      timeout(time: max_time, unit: 'MINUTES') {
        checkout scm
        VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 4
        sh """#!/bin/bash
        # conda env remove -n gluon_vision_docs -y
        set -ex
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        # conda env create -n gluon_vision_docs -f docs/build.yml
        conda env update -n gluon_vision_docs -f docs/build.yml --prune
        conda activate gluon_vision_docs
        export PYTHONPATH=\${PWD}
        env
        export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64
        git submodule update --init --recursive
        git clean -fx
        cd docs && make clean && make html
        sed -i.bak 's/33\\,150\\,243/23\\,141\\,201/g' build/html/_static/material-design-lite-1.3.0/material.blue-deep_orange.min.css;
        sed -i.bak 's/2196f3/178dc9/g' build/html/_static/sphinx_materialdesign_theme.css;
        sed -i.bak 's/pre{padding:1rem;margin:1.5rem\\s0;overflow:auto;overflow-y:hidden}/pre{padding:1rem;margin:1.5rem 0;overflow:auto;overflow-y:scroll}/g' build/html/_static/sphinx_materialdesign_theme.css

        if [[ ${env.BRANCH_NAME} == master ]]; then
            aws s3 cp s3://gluon-cv.mxnet.io/coverage.svg build/html/coverage.svg
            aws s3 sync --delete build/html/ s3://gluon-cv.mxnet.io/ --acl public-read --cache-control max-age=7200
            aws s3 cp build/html/coverage.svg s3://gluon-cv.mxnet.io/coverage.svg --acl public-read --cache-control max-age=300
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
}
