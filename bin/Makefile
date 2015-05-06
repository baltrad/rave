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
	@cp -v -f area_registry "${prefix}/bin/"
	@cp -v -f n2b "${prefix}/bin/"
	@cp -v -f n2b.sh "${prefix}/bin/"
	@cp -v -f pgf_help "${prefix}/bin/"
	@cp -v -f pgf_registry "${prefix}/bin/"
	@cp -v -f projection_registry "${prefix}/bin/"
	@cp -v -f radarcomp "${prefix}/bin/"
	@cp -v -f rave_pgf "${prefix}/bin/"
	@cp -v -f show "${prefix}/bin/"
	@cp -v -f sunpos "${prefix}/bin/"
	@cp -v -f dealias "${prefix}/bin/"
	@cp -v -f radvol "${prefix}/bin/"
	@cp -v -f odc_toolbox "${prefix}/bin/"
	@cp -v -f poo "${prefix}/bin/"
	@cp -v -f odc_hac "${prefix}/bin/"
	@cp -v -f fm12_importer "${prefix}/bin/"
	@cp -v -f wmo_station "${prefix}/bin/"
	@cp -v -f ctfilter "${prefix}/bin/"
	@cp -v -f site2D "${prefix}/bin/"
	@cp -v -f rave_pgf_logger "${prefix}/bin/"
	@cp -v -f create_tile_definition "${prefix}/bin/"
	@cp -v -f create_acrr_composites "${prefix}/bin/"


.PHONY:clean
clean: ;
	@\rm -f *~

.PHONY:distclean
distclean:
	@\rm -f *~
