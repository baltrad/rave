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
# For etc directory, only install and clean functionality
# @file
# @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
# @date 2012-03-14
###########################################################################
-include ../def.mk

CONFIG_FILES=rave_pgf_quality_registry.xml rave_pgf_queue.xml rave_pgf_registry.xml rave_tile_registry.xml

.PHONY=install
install:
	@mkdir -p "${prefix}/etc"
	@cp -v -f rave_pgf "${prefix}/etc/"
	@for i in $(CONFIG_FILES) ; do \
	    if [ ! -f ${prefix}/etc/$$i ]; then \
	      cp -v -f $$i ${prefix}/etc/; \
	    fi; \
	done

.PHONY:clean
clean: ;
	@\rm -f *~
	 
.PHONY:distclean
distclean:
	@\rm -f *~
	