#!/usr/bin/env bash

BRANCH=$(basename $1)
PR_NUMBER=$2
COMMIT_SHA=$3

EFS=/mnt/efs

mkdir -p ~/.mxnet/datasets
for f in $EFS/.mxnet/datasets/*; do
    if [ -d "$f" ]; then
        # Will not run if no directories are available
        ln -s $f ~/.mxnet/datasets/$(basename "$f")
    fi
done

python3 -m pip install sphinx>=1.5.5 sphinx-gallery sphinx_rtd_theme matplotlib Image recommonmark scipy mxtheme

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
cd docs 
make html
COMMAND_EXIT_CODE=$?
sed -i.bak 's/33\\,150\\,243/23\\,141\\,201/g' build/html/_static/material-design-lite-1.3.0/material.blue-deep_orange.min.css;
sed -i.bak 's/2196f3/178dc9/g' build/html/_static/sphinx_materialdesign_theme.css;
sed -i.bak 's/pre{padding:1rem;margin:1.5rem\\s0;overflow:auto;overflow-y:hidden}/pre{padding:1rem;margin:1.5rem 0;overflow:auto;overflow-y:scroll}/g' build/html/_static/sphinx_materialdesign_theme.css

if [[ $BRANCH == master ]]; then
	# aws s3 cp s3://gluon-cv.mxnet.io/coverage.svg build/html/coverage.svg
	aws s3 sync --delete build/html/ s3://gluoncv-ci/build_docs/master/ --acl public-read --cache-control max-age=7200
	# aws s3 cp build/html/coverage.svg s3://gluon-cv.mxnet.io/coverage.svg --acl public-read --cache-control max-age=300
	# echo "Uploaded doc to http://gluon-cv.mxnet.io"
	echo master
else
	# aws s3 cp s3://gluoncv-ci/build_docs/$PR_NUMBER/$COMMIT_SHA/coverage.svg build/html/coverage.svg
	aws s3 sync --delete build/html/ s3://gluoncv-ci/build_docs/$PR_NUMBER/$COMMIT_SHA/ --acl public-read
	# echo "Uploaded doc to http://gluon-vision-staging.s3-website-us-west-2.amazonaws.com/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/index.html"
	echo $BRANCH
fi;
exit $COMMAND_EXIT_CODE
