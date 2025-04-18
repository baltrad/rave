###########################################################################
# Copyright (C) 2011 Swedish Meteorological and Hydrological Institute, SMHI,
#
# This file is part of beamb.
#
# beamb is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# beamb is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Pub_38lic License
# along with beamb.  If not, see <http://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------
# 
# beamb make file
# @file
# @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
# @date 2011-11-14
###########################################################################
-include ../def.mk

CFLAGS= $(CCSHARED) $(OPTS) $(DEFS) -I../librave/toolbox -I../librave/pyapi  -I../librave/radvol/lib \
			-I../librave/scansun -I. $(INCLUDE_PYTHON) \
			-I$(HLHDF_INCLUDE_DIR) $(PROJ_INCLUDE_DIR) $(NUMPY_INCLUDE_DIR) \
			$(ZLIB_INCDIR) $(HDF5_INCDIR)

# Linker flags
#
LDFLAGS= -L../librave/toolbox -L../librave/pyapi -L../librave/scansun -L. \
			-L$(HLHDF_LIB_DIR) $(PROJ_LIB_DIR) $(ZLIB_LIBDIR) $(HDF5_LIBDIR)

ifeq ($(EXPAT_SUPPRESSED), no)
CFLAGS+= -I../librave/radvol/lib $(EXPAT_INCLUDE_DIR) -DRAVE_XML_SUPPORTED  
LDFLAGS+= -L../librave/radvol/lib $(EXPAT_LIB_DIR) 
EXPAT_LIBRARY=-lexpat
endif

ifeq ($(JSONC_SUPPRESSED), no)
CFLAGS+= $(JSONC_INCLUDE_DIR) -DRAVE_JSON_SUPPORTED  
LDFLAGS+= $(JSONC_LIB_DIR) 
JSONC_LIBRARY=-ljson-c
endif

ifeq ($(BUFR_SUPPRESSED), no)
CFLAGS+= $(BUFR_INCLUDE_DIR) -DRAVE_BUFR_SUPPORTED
LDFLAGS+= $(BUFR_LIB_DIR)
BUFR_LIBRARY=-lOperaBufr
endif

ifeq ($(NETCDF_SUPPRESSED), no)
CFLAGS+= $(NETCDF_INCLUDE_DIR)
LDFLAGS+= $(NETCDF_LIB_DIR)
NETCDF_LIBRARY=-lnetcdf
ifeq ($(CF_SUPPRESSED), no)
CFLAGS+= -DRAVE_CF_SUPPORTED
endif
endif

ifeq ($(GOT_PTHREAD_SUPPORT), yes)
CFLAGS+= -DPTHREAD_SUPPORTED
PTHREAD_LIBRARY=-lpthread
endif

LIBRARIES= -lravetoolbox -lravepyapi -lhlhdf -lproj  $(EXPAT_LIBRARY) $(JSONC_LIBRARY) $(BUFR_LIBRARY) $(NETCDF_LIBRARY) $(PTHREAD_LIBRARY)

# --------------------------------------------------------------------
# Fixed definitions

SOURCE_1= mean.c
OBJECTS_1= $(SOURCE_1:.c=.o)
TARGET_1= _mean.so

SOURCE_2= pyarea.c
OBJECTS_2= $(SOURCE_2:.c=.o)
TARGET_2= _area.so

SOURCE_3= pycartesian.c
OBJECTS_3= $(SOURCE_3:.c=.o)
TARGET_3= _cartesian.so

SOURCE_4= pycartesianparam.c
OBJECTS_4= $(SOURCE_4:.c=.o)
TARGET_4= _cartesianparam.so

SOURCE_5= pycartesianvolume.c
OBJECTS_5= $(SOURCE_5:.c=.o)
TARGET_5= _cartesianvolume.so

SOURCE_6= pycomposite.c
OBJECTS_6= $(SOURCE_6:.c=.o)
TARGET_6= _pycomposite.so

SOURCE_7= pycompositealgorithm.c
OBJECTS_7= $(SOURCE_7:.c=.o)
TARGET_7= _compositealgorithm.so

SOURCE_8= pydetectionrange.c
OBJECTS_8= $(SOURCE_8:.c=.o)
TARGET_8= _detectionrange.so

SOURCE_9= pypolarnav.c
OBJECTS_9= $(SOURCE_9:.c=.o)
TARGET_9= _polarnav.so

SOURCE_10= pypolarscan.c
OBJECTS_10= $(SOURCE_10:.c=.o)
TARGET_10= _polarscan.so

SOURCE_11= pypolarscanparam.c
OBJECTS_11= $(SOURCE_11:.c=.o)
TARGET_11= _polarscanparam.so

SOURCE_12= pypolarvolume.c
OBJECTS_12= $(SOURCE_12:.c=.o)
TARGET_12= _polarvolume.so

SOURCE_13= pypoocompositealgorithm.c
OBJECTS_13= $(SOURCE_13:.c=.o)
TARGET_13= _poocompositealgorithm.so

SOURCE_14= pyverticalprofile.c
OBJECTS_14= $(SOURCE_14:.c=.o)
TARGET_14= _verticalprofile.so

SOURCE_15= pyprojection.c
OBJECTS_15= $(SOURCE_15:.c=.o)
TARGET_15= _projection.so

SOURCE_16= pyradardefinition.c
OBJECTS_16= $(SOURCE_16:.c=.o)
TARGET_16= _radardef.so

SOURCE_17= pyravefield.c
OBJECTS_17= $(SOURCE_17:.c=.o)
TARGET_17= _ravefield.so

SOURCE_18= pyraveio.c
OBJECTS_18= $(SOURCE_18:.c=.o)
TARGET_18= _raveio.so

SOURCE_19= pyscansun.c
OBJECTS_19= $(SOURCE_19:.c=.o)
TARGET_19= _scansun.so

SOURCE_20= pytransform.c
OBJECTS_20= $(SOURCE_20:.c=.o)
TARGET_20= _transform.so

SOURCE_21= rave.c
OBJECTS_21= $(SOURCE_21:.c=.o)
TARGET_21= _rave.so

SOURCE_22= pyprojectionregistry.c
OBJECTS_22= $(SOURCE_22:.c=.o)
TARGET_22= _projectionregistry.so

SOURCE_23= pyarearegistry.c
OBJECTS_23= $(SOURCE_23:.c=.o)
TARGET_23= _arearegistry.so

SOURCE_24= _proj.c
OBJECTS_24= $(SOURCE_24:.c=.o)
TARGET_24= _proj.so

SOURCE_25= h5rad.c
OBJECTS_25= $(SOURCE_25:.c=.o)
TARGET_25= _h5rad.so

SOURCE_26= helpers.c
OBJECTS_26= $(SOURCE_26:.c=.o)
TARGET_26= _helpers.so

SOURCE_27= ctoc.c
OBJECTS_27= $(SOURCE_27:.c=.o)
TARGET_27= _ctoc.so

SOURCE_28= composite.c
OBJECTS_28= $(SOURCE_28:.c=.o)
TARGET_28= _composite.so

SOURCE_29= ptop.c
OBJECTS_29= $(SOURCE_29:.c=.o)
TARGET_29= _ptop.so

SOURCE_30= pyacrr.c
OBJECTS_30= $(SOURCE_30:.c=.o)
TARGET_30= _acrr.so

SOURCE_31= pydealias.c
OBJECTS_31= $(SOURCE_31:.c=.o)
TARGET_31= _dealias.so

SOURCE_32= pyradvol.c
OBJECTS_32= $(SOURCE_32:.c=.o)
TARGET_32= _radvol.so

SOURCE_33= pyodc_hac.c
OBJECTS_33= $(SOURCE_33:.c=.o)
TARGET_33= _odc_hac.so

SOURCE_34= pycartesiancomposite.c
OBJECTS_34= $(SOURCE_34:.c=.o)
TARGET_34= _cartesiancomposite.so

SOURCE_35= pyqitotal.c
OBJECTS_35= $(SOURCE_35:.c=.o)
TARGET_35= _qitotal.so

SOURCE_36= pygra.c
OBJECTS_36= $(SOURCE_36:.c=.o)
TARGET_36= _gra.so

SOURCE_37= pyctfilter.c
OBJECTS_37= $(SOURCE_37:.c=.o)
TARGET_37= _ctfilter.so

SOURCE_38= pyacqva.c
OBJECTS_38= $(SOURCE_38:.c=.o)
TARGET_38= _acqva.so

SOURCE_39= pybitmapgenerator.c
OBJECTS_39= $(SOURCE_39:.c=.o)
TARGET_39= _bitmapgenerator.so

SOURCE_40= pyravedata2d.c
OBJECTS_40= $(SOURCE_40:.c=.o)
TARGET_40= _ravedata2d.so

SOURCE_41= pylazynodelistreader.c
OBJECTS_41= $(SOURCE_41:.c=.o)
TARGET_41= _lazynodelistreader.so

SOURCE_42= pyprojectionpipeline.c
OBJECTS_42= $(SOURCE_42:.c=.o)
TARGET_42= _projectionpipeline.so

SOURCE_43= pyraveattributetable.c
OBJECTS_43= $(SOURCE_43:.c=.o)
TARGET_43= _attributetable.so

SOURCE_44= pyiocache.c
OBJECTS_44= $(SOURCE_44:.c=.o)
TARGET_44= _iocache.so

SOURCE_45= pyravelegend.c
OBJECTS_45= $(SOURCE_45:.c=.o)
TARGET_45= _ravelegend.so

SOURCE_46= pypolarnavinfo.c
OBJECTS_46= $(SOURCE_46:.c=.o)
TARGET_46= _pypolarnavinfo.so

SOURCE_47= pycompositegenerator.c
OBJECTS_47= $(SOURCE_47:.c=.o)
TARGET_47= _compositegenerator.so

SOURCE_48= pycompositegeneratorfactory.c
OBJECTS_48= $(SOURCE_48:.c=.o)
TARGET_48= _compositegeneratorfactory.so

SOURCE_49= pylegacycompositegeneratorfactory.c
OBJECTS_49= $(SOURCE_49:.c=.o)
TARGET_49= _legacycompositegeneratorfactory.so

SOURCE_50= pycompositearguments.c
OBJECTS_50= $(SOURCE_50:.c=.o)
TARGET_50= _compositearguments.so

SOURCE_51= pyacqvacompositegeneratorfactory.c
OBJECTS_51= $(SOURCE_51:.c=.o)
TARGET_51= _acqvacompositegeneratorfactory.so

SOURCE_52= pyodimsource.c
OBJECTS_52= $(SOURCE_52:.c=.o)
TARGET_52= _odimsource.so

SOURCE_53= pyodimsources.c
OBJECTS_53= $(SOURCE_53:.c=.o)
TARGET_53= _odimsources.so

SOURCE_54= pycompositefilter.c
OBJECTS_54= $(SOURCE_54:.c=.o)
TARGET_54= _compositefilter.so

SOURCE_55= pycompositefactorymanager.c
OBJECTS_55= $(SOURCE_55:.c=.o)
TARGET_55= _compositefactorymanager.so

SOURCE_56= pynearestcompositegeneratorfactory.c
OBJECTS_56= $(SOURCE_56:.c=.o)
TARGET_56= _nearestcompositegeneratorfactory.so

SOURCE_57= pyravevalue.c
OBJECTS_57= $(SOURCE_57:.c=.o)
TARGET_57= _ravevalue.so

SOURCE_58= pyraveproperties.c
OBJECTS_58= $(SOURCE_58:.c=.o)
TARGET_58= _raveproperties.so

TARGETS=$(TARGET_1) $(TARGET_2) $(TARGET_3) $(TARGET_4) $(TARGET_5) $(TARGET_6) \
		$(TARGET_7) $(TARGET_8) $(TARGET_9) $(TARGET_10) $(TARGET_11) $(TARGET_12) $(TARGET_13) \
		$(TARGET_14) $(TARGET_15) $(TARGET_16) $(TARGET_17) $(TARGET_18) $(TARGET_19) \
		$(TARGET_20) $(TARGET_21) $(TARGET_24) $(TARGET_25) \
		$(TARGET_26) $(TARGET_27) $(TARGET_28) $(TARGET_29) $(TARGET_30) $(TARGET_31) \
		$(TARGET_33) $(TARGET_34) $(TARGET_35) $(TARGET_36) $(TARGET_37) $(TARGET_38) \
		$(TARGET_39) $(TARGET_40) $(TARGET_41) $(TARGET_42) $(TARGET_43) $(TARGET_44) $(TARGET_45) \
		$(TARGET_46) $(TARGET_47) $(TARGET_48) $(TARGET_49) $(TARGET_50) $(TARGET_51) \
		$(TARGET_52) $(TARGET_53) $(TARGET_54) $(TARGET_55) $(TARGET_56) $(TARGET_57) \
		$(TARGET_58)

INSTALL_HEADERS= pyarea.h \
				 pycartesian.h \
				 pycartesianparam.h \
				 pycartesianvolume.h \
				 pycompositealgorithm.h \
				 pycomposite.h \
				 pydetectionrange.h \
				 pypolarnav.h \
				 pypolarscan.h \
				 pypolarscanparam.h \
				 pypolarvolume.h \
				 pyprojection.h \
				 pyradardefinition.h \
				 pyrave_debug.h \
				 pyravefield.h \
				 pyraveio.h \
				 pytransform.h \
				 pyacrr.h \
				 pyverticalprofile.h \
				 pycartesiancomposite.h \
				 pycartesiangenerator.h \
				 pyqitotal.h \
				 pygra.h \
				 pybitmapgenerator.h \
				 pyravedata2d.h \
				 pylazynodelistreader.h \
				 pyprojectionpipeline.h \
				 pyraveattributetable.h \
				 pyacqva.h \
				 pyiocache.h \
				 pyravelegend.h \
				 pypolarnavinfo.h \
				 pycompositegenerator.h \
				 pycompositegeneratorfactory.h \
				 pycompositearguments.h \
				 pyodimsource.h \
				 pyodimsources.h \
				 pycompositefilter.h \
				 pycompositefactorymanager.h \
				 pyravevalue.h \
				 pyraveproperties.h

ifeq ($(EXPAT_SUPPRESSED), no)
TARGETS += $(TARGET_22) $(TARGET_23) $(TARGET_32)
INSTALL_HEADERS += pyprojectionregistry.h pyarearegistry.h
endif

# We reset TARGETS and INSTALL_HEADERS if we shouldn't compile for python. That way we don't need to destroy anything
ifeq ($(COMPILE_FOR_PYTHON), no)
TARGETS=
INSTALL_HEADERS=
endif


MAKECDEPEND=$(CC) -MM $(CFLAGS) -MT '$(@D)/$(@F)' -o $(DF).d $<

DEPDIR=.dep
DF=$(DEPDIR)/$(*F)

# Ensures that the .dep directory exists
.PHONY=$(DEPDIR)
$(DEPDIR):
	+@[ -d $@ ] || mkdir -p $@

# And the rest of the make file targets
#
.PHONY=all
all:		$(TARGETS)

$(TARGET_1): $(DEPDIR) $(OBJECTS_1) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_1) $(LDFLAGS) $(LIBRARIES)

$(TARGET_2): $(DEPDIR) $(OBJECTS_2) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_2) $(LDFLAGS) $(LIBRARIES)

$(TARGET_3): $(DEPDIR) $(OBJECTS_3) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_3) $(LDFLAGS) $(LIBRARIES)

$(TARGET_4): $(DEPDIR) $(OBJECTS_4) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_4) $(LDFLAGS) $(LIBRARIES)

$(TARGET_5): $(DEPDIR) $(OBJECTS_5) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_5) $(LDFLAGS) $(LIBRARIES)

$(TARGET_6): $(DEPDIR) $(OBJECTS_6) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_6) $(LDFLAGS) $(LIBRARIES)

$(TARGET_7): $(DEPDIR) $(OBJECTS_7) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_7) $(LDFLAGS) $(LIBRARIES)

$(TARGET_8): $(DEPDIR) $(OBJECTS_8) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_8) $(LDFLAGS) $(LIBRARIES)

$(TARGET_9): $(DEPDIR) $(OBJECTS_9) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_9) $(LDFLAGS) $(LIBRARIES)

$(TARGET_10): $(DEPDIR) $(OBJECTS_10) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_10) $(LDFLAGS) $(LIBRARIES)

$(TARGET_11): $(DEPDIR) $(OBJECTS_11) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_11) $(LDFLAGS) $(LIBRARIES)

$(TARGET_12): $(DEPDIR) $(OBJECTS_12) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_12) $(LDFLAGS) $(LIBRARIES)

$(TARGET_13): $(DEPDIR) $(OBJECTS_13) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_13) $(LDFLAGS) $(LIBRARIES)

$(TARGET_14): $(DEPDIR) $(OBJECTS_14) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_14) $(LDFLAGS) $(LIBRARIES)

$(TARGET_15): $(DEPDIR) $(OBJECTS_15) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_15) $(LDFLAGS) $(LIBRARIES)

$(TARGET_16): $(DEPDIR) $(OBJECTS_16) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_16) $(LDFLAGS) $(LIBRARIES)

$(TARGET_17): $(DEPDIR) $(OBJECTS_17) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_17) $(LDFLAGS) $(LIBRARIES)

$(TARGET_18): $(DEPDIR) $(OBJECTS_18) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_18) $(LDFLAGS) $(LIBRARIES)

$(TARGET_19): $(DEPDIR) $(OBJECTS_19) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so ../librave/scansun/libravescansun.so
	$(LDSHARED) -o $@ $(OBJECTS_19) $(LDFLAGS) -lravescansun $(LIBRARIES)

$(TARGET_20): $(DEPDIR) $(OBJECTS_20) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_20) $(LDFLAGS) $(LIBRARIES)

$(TARGET_21): $(DEPDIR) $(OBJECTS_21) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_21) $(LDFLAGS) $(LIBRARIES)

$(TARGET_22): $(DEPDIR) $(OBJECTS_22) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_22) $(LDFLAGS) $(LIBRARIES)

$(TARGET_23): $(DEPDIR) $(OBJECTS_23) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_23) $(LDFLAGS) $(LIBRARIES)

$(TARGET_24): $(DEPDIR) $(OBJECTS_24) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_24) $(LDFLAGS) $(LIBRARIES)

$(TARGET_25): $(DEPDIR) $(OBJECTS_25) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_25) $(LDFLAGS) $(LIBRARIES)

$(TARGET_26): $(DEPDIR) $(OBJECTS_26) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_26) $(LDFLAGS) $(LIBRARIES)

$(TARGET_27): $(DEPDIR) $(OBJECTS_27) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_27) $(LDFLAGS) $(LIBRARIES)

$(TARGET_28): $(DEPDIR) $(OBJECTS_28) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_28) $(LDFLAGS) $(LIBRARIES)

$(TARGET_29): $(DEPDIR) $(OBJECTS_29) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_29) $(LDFLAGS) $(LIBRARIES)

$(TARGET_30): $(DEPDIR) $(OBJECTS_30) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_30) $(LDFLAGS) $(LIBRARIES)

$(TARGET_31): $(DEPDIR) $(OBJECTS_31) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_31) $(LDFLAGS) $(LIBRARIES)

$(TARGET_32): $(DEPDIR) $(OBJECTS_32) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so ../librave/radvol/lib/libradvol.so
	$(LDSHARED) -o $@ $(OBJECTS_32) $(LDFLAGS) -lradvol $(LIBRARIES)

$(TARGET_33): $(DEPDIR) $(OBJECTS_33) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_33) $(LDFLAGS) $(LIBRARIES)

$(TARGET_34): $(DEPDIR) $(OBJECTS_34) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_34) $(LDFLAGS) $(LIBRARIES)

$(TARGET_35): $(DEPDIR) $(OBJECTS_35) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_35) $(LDFLAGS) $(LIBRARIES)

$(TARGET_36): $(DEPDIR) $(OBJECTS_36) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_36) $(LDFLAGS) $(LIBRARIES)

$(TARGET_37): $(DEPDIR) $(OBJECTS_37) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_37) $(LDFLAGS) $(LIBRARIES)

$(TARGET_38): $(DEPDIR) $(OBJECTS_38) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_38) $(LDFLAGS) $(LIBRARIES)

$(TARGET_39): $(DEPDIR) $(OBJECTS_39) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_39) $(LDFLAGS) $(LIBRARIES)

$(TARGET_40): $(DEPDIR) $(OBJECTS_40) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_40) $(LDFLAGS) $(LIBRARIES)

$(TARGET_41): $(DEPDIR) $(OBJECTS_41) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_41) $(LDFLAGS) $(LIBRARIES)

$(TARGET_42): $(DEPDIR) $(OBJECTS_42) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_42) $(LDFLAGS) $(LIBRARIES)

$(TARGET_43): $(DEPDIR) $(OBJECTS_43) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_43) $(LDFLAGS) $(LIBRARIES)

$(TARGET_44): $(DEPDIR) $(OBJECTS_44) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_44) $(LDFLAGS) $(LIBRARIES)

$(TARGET_45): $(DEPDIR) $(OBJECTS_45) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_45) $(LDFLAGS) $(LIBRARIES)

$(TARGET_46): $(DEPDIR) $(OBJECTS_46) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_46) $(LDFLAGS) $(LIBRARIES)

$(TARGET_47): $(DEPDIR) $(OBJECTS_47) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_47) $(LDFLAGS) $(LIBRARIES)

$(TARGET_48): $(DEPDIR) $(OBJECTS_48) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_48) $(LDFLAGS) $(LIBRARIES)

$(TARGET_49): $(DEPDIR) $(OBJECTS_49) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_49) $(LDFLAGS) $(LIBRARIES)

$(TARGET_50): $(DEPDIR) $(OBJECTS_50) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_50) $(LDFLAGS) $(LIBRARIES)

$(TARGET_51): $(DEPDIR) $(OBJECTS_51) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_51) $(LDFLAGS) $(LIBRARIES)

$(TARGET_52): $(DEPDIR) $(OBJECTS_52) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_52) $(LDFLAGS) $(LIBRARIES)

$(TARGET_53): $(DEPDIR) $(OBJECTS_53) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_53) $(LDFLAGS) $(LIBRARIES)

$(TARGET_54): $(DEPDIR) $(OBJECTS_54) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_54) $(LDFLAGS) $(LIBRARIES)

$(TARGET_55): $(DEPDIR) $(OBJECTS_55) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_55) $(LDFLAGS) $(LIBRARIES)

$(TARGET_56): $(DEPDIR) $(OBJECTS_56) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_56) $(LDFLAGS) $(LIBRARIES)

$(TARGET_57): $(DEPDIR) $(OBJECTS_57) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_57) $(LDFLAGS) $(LIBRARIES)

$(TARGET_58): $(DEPDIR) $(OBJECTS_58) ../librave/toolbox/libravetoolbox.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_58) $(LDFLAGS) $(LIBRARIES)

ifeq ($(COMPILE_FOR_PYTHON), yes)
.PHONY=install
install:
	@mkdir -p ${DESTDIR}${prefix}/Lib
	@mkdir -p ${DESTDIR}${prefix}/include/python
	@for i in $(TARGETS) ; \
	do \
		"$(HLHDF_INSTALL_BIN)" -f -o -m755 -C $$i "${DESTDIR}$(prefix)/Lib/$$i"; \
	done
	@for i in $(INSTALL_HEADERS) ; \
	do \
		"$(HLHDF_INSTALL_BIN)" -f -o -m644 -C $$i "${DESTDIR}$(prefix)/include/python/$$i"; \
	done
else
.PHONY=install
install:
	:
endif

.PHONY=clean
clean:
	@\rm -f *.o core *~
	@\rm -fr $(DEPDIR)

.PHONY=distclean		 
distclean:	clean
	@\rm -f $(TARGETS)

# --------------------------------------------------------------------
# Rules

# Contains dependency generation as well, so if you are not using
# gcc, comment out everything until the $(CC) statement.
%.o : %.c
	@$(MAKECDEPEND); \
	cp $(DF).d $(DF).P; \
	sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(DF).d >> $(DF).P; \
	\rm -f $(DF).d
	$(CC) -c $(CFLAGS) $< -o $@

# NOTE! This ensures that the dependencies are setup at the right time so this should not be moved
-include $(SOURCE_1:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_2:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_3:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_4:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_5:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_6:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_7:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_8:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_9:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_10:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_11:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_12:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_13:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_14:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_15:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_16:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_17:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_18:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_19:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_20:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_21:%.c=$(DEPDIR)/%.P)
ifeq ($(EXPAT_SUPPRESSED), no)
-include $(SOURCE_22:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_23:%.c=$(DEPDIR)/%.P)
endif
-include $(SOURCE_24:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_25:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_26:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_27:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_28:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_29:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_30:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_31:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_33:%.c=$(DEPDIR)/%.P)
ifeq ($(EXPAT_SUPPRESSED), no)
-include $(SOURCE_32:%.c=$(DEPDIR)/%.P)
endif
-include $(SOURCE_34:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_35:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_36:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_37:%.c=$(DEPDIR)/%.P)
#-include $(SOURCE_38:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_39:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_40:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_41:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_42:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_43:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_44:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_45:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_46:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_47:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_48:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_49:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_50:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_51:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_52:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_53:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_54:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_55:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_56:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_57:%.c=$(DEPDIR)/%.P)
-include $(SOURCE_58:%.c=$(DEPDIR)/%.P)