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

stage("Docs") {
  node {
    ws('workspace/gluon-vision-docs') {
      checkout scm
      sh """#!/bin/bash
      set -e
      set -x
      conda env update -f docs/build.yml
      source activate gluon_vision_docs
      export PYTHONPATH=\${PWD}
      env
      export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64
      cd docs && make html
      """
      
      if (env.BRANCH_NAME == "master") {
        def job = 'master-${env.BUILD_NUMBER}'
        def dst = 's3://gluon-vision.mxnet.io/'
        def url = 'http://gluon-vision.mxnet.io'
      } else {
        def job = '${env.BRANCH_NAME}-${env.BUILD_NUMBER}'
        def dst = 's3://gluon-vision-staging/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/'
        def url = 'http://gluon-vision-staging.s3-website-us-west-2.amazonaws.com/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/index.html'
      }
      
      retry (5) {
        try {
          sh "aws s3 sync --delete docs/build/html/ ${dst} --acl public-read"
        } catch (exc) {
          sh "aws s3 rm ${dst} --recursive"
          error "Failed to upload document"
        }
      }
      pullRequest.comment("Job ${job} is done. Docs are uploaded to ${url}")                          
    }
  }
}

