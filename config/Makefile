###########################################################################
# Copyright (C) 2012 Swedish Meteorological and Hydrological Institute, SMHI,
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
# For config directory, only install and clean functionality
# @file
# @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
# @date 2012-03-14
###########################################################################
-include ../def.mk

CONFIG_FILES=`ls -1 *.xml`
CONFIG_FILES+= `ls -1 *.json`

.PHONY=install
install:
  echo "$(CONFIG_FILES)"
	@mkdir -p "${DESTDIR}${prefix}/config"
	@for i in $(CONFIG_FILES) ; do \
	    if [ ! -f ${DESTDIR}${prefix}/config/$$i ]; then \
	      cp -v -f $$i ${DESTDIR}${prefix}/config/; \
	    fi; \
	done
	if [ -f ${DESTDIR}${prefix}/config/odim_quantities.xml ]; then \
	  \rm -f ${DESTDIR}${prefix}/config/odim_quantities.xml; \
	fi
	cp -v -f odim_quantities.xml ${DESTDIR}${prefix}/config/;
	 

.PHONY:clean
clean: ;
	@\rm -f *~
	 
.PHONY:distclean
distclean:
	@\rm -f *~
	
