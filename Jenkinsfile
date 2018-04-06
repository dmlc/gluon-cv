stage("LINT") {
  node {
    ws('workspace/gluon-vision-lint') {
      checkout scm
      sh "export PATH=/var/lib/jenkins/miniconda3/bin:$PATH"
      echo "$PATH"
      sh "conda env update -f tests/pylint.yml"
      sh "source activate gluon_vision_pylint"
      sh "make pylint"
    }
  }
}

stage("Docs") {
  node {
    ws('workspace/gluon-vision-docs') {
      checkout scm
      sh "export PATH=/var/lib/jenkins/miniconda3/bin:$PATH"
      sh "conda env update -f docs/build.yml"
      sh "source activate gluon_vision_docs"
      sh "python setup.py install"
      sh "cd docs && make html"
      sh "aws s3 sync --delete build/html/ s3://gluon-vision.mxnet.io/ --acl public-read"
    }
  }
}
