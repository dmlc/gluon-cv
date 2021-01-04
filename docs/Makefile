# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = GluonCV
SOURCEDIR     = .
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean

clean:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	wget -O _static/classification_throughputs.html https://zhiz-datastore.s3-us-west-2.amazonaws.com/gluoncv-throughput-benchmarks/classification_throughputs.html
	wget -O _static/detection_throughputs.html https://zhiz-datastore.s3-us-west-2.amazonaws.com/gluoncv-throughput-benchmarks/detection_throughputs.html
	wget -O _static/detection_coco_per_class.html https://zhiz-datastore.s3-us-west-2.amazonaws.com/gluoncv-throughput-benchmarks/detection_coco_per_class.html
	wget -O _static/pose_throughputs.html https://zhiz-datastore.s3-us-west-2.amazonaws.com/gluoncv-throughput-benchmarks/pose_throughputs.html
	wget -O _static/semantic_segmentation_throughputs.html https://zhiz-datastore.s3-us-west-2.amazonaws.com/gluoncv-throughput-benchmarks/semantic_segmentation_throughputs.html
	wget -O _static/ar_throughputs.html https://zhiz-datastore.s3-us-west-2.amazonaws.com/gluoncv-throughput-benchmarks/ar_throughputs.html
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
