###########################################################################
# Copyright (C) 2011 Swedish Meteorological and Hydrological Institute, SMHI,
#
# This file is part of Radvol-QC.
#
# Radvol-QC is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Radvol-QC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with RAVE.  If not, see <http://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------
# 
# testRadvolQC make file
# @file
# @author
# @date
###########################################################################
-include ../../../def.mk

# c flags, use rave suggested ones
#
CFLAGS= -I. $(RAVE_MODULE_CFLAGS) -I../lib -ICUnit-2.1-2/include
LDFLAGS= $(RAVE_MODULE_LDFLAGS) -L../lib -LCUnit-2.1-2/lib
LIBRARIES= $(RAVE_MODULE_LIBRARIES) -lradvol -lcunit
export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:../lib:CUnit-2.1-2/lib

# --------------------------------------------------------------------
# Fixed definitions

SOURCES= testRadvol.c testRadvolAtt.c testRadvolBroad.c testRadvolSpeck.c testRadvolSpike.c testRadvolQC.c 

TARGET= testRadvolQC

.PHONY:test
test:
	@\tar -xvjf CUnit-2.1-2-src.tar.bz2	
	@\cd CUnit-2.1-2 && ./configure --prefix=`pwd`
	@\cd CUnit-2.1-2/CUnit && make
	@\cd CUnit-2.1-2/CUnit && make install
	$(CC) $(LDFLAGS) $(CFLAGS)  $(SOURCES) -o $(TARGET) $(LIBRARIES) 
	./testRadvolQC
	rm -fr CUnit-2.1-2

.PHONY=clean
clean:
	@\rm -f $(TARGET) *.o core *~
	@\rm -fr CUnit-2.1-2

.PHONY=distclean		 
distclean:	clean


