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
	@mkdir -p "${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} area_registry "${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} odim_injector "${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} odim_injector.sh "${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} pgf_help "${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} pgf_registry "${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} projection_registry "${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} radarcomp "${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} rave_pgf "${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} show "${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} sunpos "${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} dealias "${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} radvol "${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} odc_toolbox "${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} poo "${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} odc_hac "${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} fm12_importer "${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} wmo_station "${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} ctfilter "${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} rave_pgf_logger "${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} create_tile_definition "${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} create_acrr_composites "${prefix}/bin"
	@./fix_shebang.sh ${PYTHON_BIN} merge_files "${prefix}/bin"
	
.PHONY:clean
clean: ;
	@\rm -f *~

.PHONY:distclean
distclean:
	@\rm -f *~
