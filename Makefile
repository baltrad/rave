###########################################################################
# Copyright (C) 2009 Swedish Meteorological and Hydrological Institute, SMHI,
#
# This file is part of RAVE.
#
# RAVE is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# RAVE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with RAVE.  If not, see <http://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------
# 
# Main build file
# @file
# @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
# @date 2009-12-10
###########################################################################

SETUP = python setup.py

all: build

def.mk:
	$(MAKE) -C librave def.mk

librave/transform/libravetransform.so librave/pyapi/libravepyapi.so: def.mk
	$(MAKE) -C librave

.PHONY:test
test:
	@chmod +x ./tools/test_rave.sh
	@./tools/test_rave.sh

.PHONY:alltest
alltest:
	@chmod +x ./tools/test_rave.sh
	@./tools/test_rave.sh alltest
	
.PHONY:doc
doc:
	$(MAKE) -C doxygen doc

.PHONY:build
build: librave/transform/libravetransform.so librave/pyapi/libravepyapi.so
	@\rm -fr build
	$(SETUP) build

.PHONY:install
install:
		$(MAKE) -C librave install
		$(SETUP) install

.PHONY:uninstall
uninstall:
		rm -rf $(RAVEROOT)

.PHONY:clean
clean:
		$(SETUP) clean
		$(MAKE) -C doxygen clean
		$(MAKE) -C librave clean
		$(MAKE) -C test/pytest clean

.PHONY:distclean
distclean:
		$(MAKE) -C librave distclean
		$(MAKE) -C doxygen distclean
		$(MAKE) -C test/pytest distclean
		@\rm -fr build
		@\rm -f *~
