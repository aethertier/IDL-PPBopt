PACKAGE_NAME = IDLPPBopt

.PHONY: build install uninstall clean

build:
	python -m build

install:
	pip install dist/$(PACKAGE_NAME)-*.tar.gz

uninstall:
	pip uninstall -y $(PACKAGE_NAME)

clean:
	rm -rf dist Code/$(PACKAGE_NAME).egg-info
