#!/bin/bash
#
# Build and publish all docs
set -x
set -e

conda env update -f docs/build.yml
source activate gluon_vision_docs

python setup.py install

cd docs

make html

aws s3 sync --delete build/html/ s3://gluon-vision.mxnet.io/ --acl public-read
