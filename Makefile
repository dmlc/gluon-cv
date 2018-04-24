ROOTDIR = $(CURDIR)

lint: cpplint pylint

cpplint:
				tests/lint.py gluoncv cpp src

pylint:
				pylint --rcfile=$(ROOTDIR)/tests/pylintrc --ignore-patterns=".*\.so$$,.*\.dll$$,.*\.dylib$$" gluoncv

doc: docs

clean_docs:
						make -C docs clean
