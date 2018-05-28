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
# Install target for root directory so that the installation doesn't trigger
# automatically etc. 
# @file
# @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
# @date 2009-12-10
###########################################################################
-include def.mk

def.mk:
	+[ -f $@ ] || $(error You need to run ./configure)

install: def.mk
	@mkdir -p "${DESTDIR}${prefix}"
	@cp -v -f rave.xbm "${DESTDIR}${prefix}/"
	@cp -v -f COPYING "${DESTDIR}${prefix}/"
	@cp -v -f COPYING.LESSER "${DESTDIR}${prefix}/"
	@cp -v -f LICENSE "${DESTDIR}${prefix}/"
	@mkdir -p "${DESTDIR}${prefix}/mkf"
	@cp -v -f def.mk "${DESTDIR}${prefix}/mkf/"
		
