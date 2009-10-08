# Makefile for RAVE.
# Only a simple wrapper around distutils, except for compiling librave 
# which is a wrapped conventional build.

SETUP = python setup.py

all: build

def.mk:
	$(MAKE) -C librave def.mk

librave/transform/libravetransform.so librave/pyapi/libravepyapi.so: def.mk
	$(MAKE) -C librave

check:
	@chmod +x ./tools/test_rave.sh
	@./tools/test_rave.sh

	
.PHONY=docs
docs:
	$(MAKE) -C doxygen docs

build: librave/transform/libravetransform.so librave/pyapi/libravepyapi.so
	@\rm -fr build
	$(SETUP) build

install:
		$(MAKE) -C librave install
		$(SETUP) install

uninstall:
		rm -rf $(RAVEROOT)

clean:
		$(SETUP) clean
		$(MAKE) -C doxygen clean
		$(MAKE) -C librave clean
		
distclean:
		$(MAKE) -C librave distclean
		$(MAKE) -C doxygen distclean
		@\rm -fr build
		@\rm -f *~
		
