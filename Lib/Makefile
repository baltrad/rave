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
# For Lib directory, only install and clean functionality
# @file
# @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
# @date 2012-03-12
###########################################################################
-include ../def.mk

ifeq ($(COMPILE_FOR_PYTHON), yes)
.PHONY=install
install:
	@mkdir -p "${DESTDIR}${prefix}/Lib"
	@cp -v -f *.py "${DESTDIR}${prefix}/Lib/"
	@mkdir -p "${DESTDIR}${prefix}/Lib/gadjust"
	@cp -v -f gadjust/*.py "${DESTDIR}${prefix}/Lib/gadjust/"
	@mkdir -p "${DESTDIR}${prefix}/Lib/ravemigrate/versions"
	@cp -v -f ravemigrate/*.py "${DESTDIR}${prefix}/Lib/ravemigrate/"
	@cp -v -f ravemigrate/migrate.cfg "${DESTDIR}${prefix}/Lib/ravemigrate/"
	@cp -v -f ravemigrate/versions/*.py "${DESTDIR}${prefix}/Lib/ravemigrate/versions/"
	@\rm -fr "${prefix}/Lib/migrate" # We can't have directory "migrate" since it conflicts with sqlalchemy migrate. If rave would have been separate package then there wouldn't have been any problems
	@./make_script_update_config.sh "${DESTDIR}${prefix}/Lib/rave_defines.py"
	@mkdir -p "${DESTDIR}${prefix}/etc"
	@-mkdir -p "${DESTDIR}$(SITEPACK_PYTHON)"
	@echo "$(prefix)/Lib" > "${DESTDIR}${prefix}/etc/rave.pth"
	@-echo "$(prefix)/Lib" > "${DESTDIR}$(SITEPACK_PYTHON)/rave.pth"
else
.PHONY=install
install:
	:
endif

.PHONY:clean
clean: ;
	@\rm -f *~
	 
.PHONY:distclean
distclean:
	@\rm -f *~
