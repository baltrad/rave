###########################################################################
# Copyright (C) 2011 Swedish Meteorological and Hydrological Institute, SMHI,
#
# This file is part of radvol.
#
# radvol is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# radvol is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with radvol.  If not, see <http://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------
# 
# Main build file
# @file
# @author 
# @date 
###########################################################################
-include ../../def.mk

.PHONY:all
all: build

.PHONY:build 
build:
	$(MAKE) -C lib

.PHONY:install
install:
	$(MAKE) -C lib install
	@echo "########################################################################"
	@echo "To run RADVOL test binaries you may need to update your LD_LIBRARY_PATH"
	@echo "########################################################################"

.PHONY:doc
doc:
	$(MAKE) -C doxygen doc

.PHONY:test
test:
	$(MAKE) -C test

.PHONY:clean
clean:
	$(MAKE) -C lib clean
	$(MAKE) -C doxygen clean
	$(MAKE) -C test clean

.PHONY:distclean
distclean:
	$(MAKE) -C lib distclean
	$(MAKE) -C doxygen distclean
	$(MAKE) -C test distclean
	@\rm -f *~ config.log config.status def.mk
