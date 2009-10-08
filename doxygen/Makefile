############################################################
# Makefile for generating the documentation
# 
# Author(s):	Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
#
# Date:		2009-06-11
############################################################
all:

.PHONY=docs
docs:
	@doxygen rave_doxygen.cfg

.PHONY=clean
clean:
	@\rm -f *.o
	@\rm -f *~

.PHONY=distclean
distclean:
	@\rm -f *.o
	@\rm -f *~ core
	@\rm -fr doxygen

distribution:

install:

