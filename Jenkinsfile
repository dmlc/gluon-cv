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

environment {
  if (env.BRANCH_NAME == "master") {
    JOB = 'master-${env.BUILD_NUMBER}'
    DST = 's3://gluon-vision.mxnet.io/'
    URL = 'http://gluon-vision.mxnet.io'    
  } else {
    JOB = '${env.BRANCH_NAME}-${env.BUILD_NUMBER}'
    DST = 's3://gluon-vision-staging/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/'
    URL = 'http://gluon-vision-staging.s3-website-us-west-2.amazonaws.com/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/index.html'
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
         
      retry (5) {
        try {
          sh "aws s3 sync --delete docs/build/html/ ${DST} --acl public-read"
        } catch (exc) {
          sh "aws s3 rm ${DST} --recursive"
          error "Failed to upload document"
        }
      }
      pullRequest.comment("Job ${JOB} is done. Docs are uploaded to ${URL}")                          
    }
  }
}

