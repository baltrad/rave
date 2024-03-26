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
# For bin directory, only install and clean functionality
# @file
# @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
# @date 2012-03-14
###########################################################################
-include ../def.mk

.PHONY=install
install:
	@mkdir -p "${DESTDIR}${prefix}/bin"
ifeq ($(COMPILE_FOR_PYTHON), yes)	
	@./fix_shebang.sh ${PYTHON_BIN} area_registry "${DESTDIR}${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} odim_injector "${DESTDIR}${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} odim_injector.sh "${DESTDIR}${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} pgf_help "${DESTDIR}${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} pgf_registry "${DESTDIR}${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} projection_registry "${DESTDIR}${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} radarcomp "${DESTDIR}${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} rave_pgf "${DESTDIR}${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} show "${DESTDIR}${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} sunpos "${DESTDIR}${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} ec_drqc "${DESTDIR}${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} dealias "${DESTDIR}${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} radvol "${DESTDIR}${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} odc_toolbox "${DESTDIR}${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} poo "${DESTDIR}${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} odc_hac "${DESTDIR}${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} fm12_importer "${DESTDIR}${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} wmo_station "${DESTDIR}${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} ctfilter "${DESTDIR}${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} rave_pgf_logger "${DESTDIR}${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} create_tile_definition "${DESTDIR}${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} create_acrr_composites "${DESTDIR}${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} merge_files "${DESTDIR}${prefix}/bin"
endif	
	
.PHONY:clean
clean: ;
	@\rm -f *~

.PHONY:distclean
distclean:
	@\rm -f *~
