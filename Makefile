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
all: modules

.PHONY:librave
librave: def.mk
	$(MAKE) -C librave
	
.PHONY:modules
modules: librave
	$(MAKE) -C modules

def.mk: def.mk.in configure
	sh makelibs

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

.PHONY:install
install:
	$(MAKE) -C librave install
	$(MAKE) -C modules install
	$(MAKE) -C Lib install

.PHONY:uninstall
uninstall:
	rm -rf $(RAVEROOT)

.PHONY:clean
clean:
	$(MAKE) -C doxygen clean
	$(MAKE) -C librave clean
	$(MAKE) -C modules clean
	$(MAKE) -C test/pytest clean
	$(MAKE) -C Lib clean

.PHONY:distclean
distclean:
	$(MAKE) -C modules distclean
	$(MAKE) -C librave distclean
	$(MAKE) -C doxygen distclean
	$(MAKE) -C test/pytest distclean
	$(MAKE) -C Lib distclean
	@\rm -fr build
	@\rm -f *~ def.mk config.log config.status
