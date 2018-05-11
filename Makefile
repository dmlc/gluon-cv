ROOTDIR = $(CURDIR)

lint: cpplint pylint

cpplint:
				tests/lint.py gluoncv cpp src

pylint:
				pylint --rcfile=$(ROOTDIR)/tests/pylintrc --ignore-patterns=".*\.so$$,.*\.dll$$,.*\.dylib$$" gluoncv

doc: docs

clean: clean_build

clean_docs:
						make -C docs clean

clean_build:
						rm -rf dist gluoncv.egg-info build | true
