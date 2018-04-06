stage("LINT") {
  node {
    ws('workspace/gluon-vision-lint') {
      checkout scm
      sh "pip install pylint"
      sh "make pylint"
    }
  }
}

stage("Docs") {
  node {
    ws('workspace/gluon-vision-docs') {
      checkout scm
      sh "bash docs/build_docs.sh"
    }
  }
}
