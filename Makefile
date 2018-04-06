ROOTDIR = $(CURDIR)

lint: cpplint pylint

cpplint:
				tests/lint.py gluonvision cpp src

pylint:
				pylint --rcfile=$(ROOTDIR)/tests/pylintrc --ignore-patterns=".*\.so$$,.*\.dll$$,.*\.dylib$$" gluonvision

doc: docs

clean_docs:
						make -C docs clean
