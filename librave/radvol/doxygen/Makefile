############################################################
# Makefile for generating the documentation
# 
# Author(s):	Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
#
# Date:		
############################################################
all:

.PHONY=doc
doc:
	@doxygen radvol_doxygen.cfg

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

