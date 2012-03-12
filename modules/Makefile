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
# You should have received a copy of the GNU Lesser General Public License
# along with beamb.  If not, see <http://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------
# 
# beamb make file
# @file
# @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
# @date 2011-11-14
###########################################################################
-include ../def.mk

CFLAGS= $(CCSHARED) $(OPTS) $(DEFS) -I../librave/transform -I../librave/pyapi \
			-I../librave/scansun -I. $(INCLUDE_PYTHON) \
			-I$(HLHDF_INCLUDE_DIR) $(NUMPY_INCLUDE_DIR) \
			$(ZLIB_INCDIR) $(HDF5_INCDIR) $(PROJ_INCLUDE_DIR)

# Linker flags
#
LDFLAGS= -L../librave/transform -L../librave/pyapi -L../librave/scansun -L. \
			-L$(HLHDF_LIB_DIR) $(ZLIB_LIBDIR) $(HDF5_LIBDIR) $(PROJ_LIB_DIR)

ifeq ($(EXPAT_SUPPRESSED), no)
CFLAGS+= $(EXPAT_INCLUDE_DIR) -DRAVE_XML_SUPPORTED
LDFLAGS+= $(EXPAT_LIB_DIR)
EXPAT_LIBRARY=-lexpat
endif

ifeq ($(BUFR_SUPPRESSED), no)
CFLAGS+= $(BUFR_INCLUDE_DIR) -DRAVE_BUFR_SUPPORTED
LDFLAGS+= $(BUFR_LIB_DIR)
BUFR_LIBRARY=-lOperaBufr
endif

ifeq ($(GOT_PTHREAD_SUPPORT), yes)
CFLAGS+= -DPTHREAD_SUPPORTED
PTHREAD_LIBRARY=-lpthread
endif

LIBRARIES= -lravetransform -lravepyapi -lhlhdf -lproj  $(EXPAT_LIBRARY) $(BUFR_LIBRARY) $(PTHREAD_LIBRARY)

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

#SOURCE_14= pyxxx.c
#OBJECTS_14= $(SOURCE_14:.c=.o)
#TARGET_14= _xxx.so

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

TARGETS=$(TARGET_1) $(TARGET_2) $(TARGET_3) $(TARGET_4) $(TARGET_5) $(TARGET_6) \
		$(TARGET_7) $(TARGET_8) $(TARGET_9) $(TARGET_10) $(TARGET_11) $(TARGET_12) \
		$(TARGET_13) $(TARGET_15) $(TARGET_16) $(TARGET_17) $(TARGET_18) \
		$(TARGET_19) $(TARGET_20) $(TARGET_21) $(TARGET_24) \
		$(TARGET_25) $(TARGET_26) $(TARGET_27) $(TARGET_28) $(TARGET_29)
	
ifeq ($(EXPAT_SUPPRESSED), no)
TARGETS += $(TARGET_22) $(TARGET_23)    
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

$(TARGET_1): $(DEPDIR) $(OBJECTS_1) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_1) $(LDFLAGS) $(LIBRARIES)
	
$(TARGET_2): $(DEPDIR) $(OBJECTS_2) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_2) $(LDFLAGS) $(LIBRARIES)
	
$(TARGET_3): $(DEPDIR) $(OBJECTS_3) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_3) $(LDFLAGS) $(LIBRARIES)
	
$(TARGET_4): $(DEPDIR) $(OBJECTS_4) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_4) $(LDFLAGS) $(LIBRARIES)
	
$(TARGET_5): $(DEPDIR) $(OBJECTS_5) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_5) $(LDFLAGS) $(LIBRARIES)
	
$(TARGET_6): $(DEPDIR) $(OBJECTS_6) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_6) $(LDFLAGS) $(LIBRARIES)
	
$(TARGET_7): $(DEPDIR) $(OBJECTS_7) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_7) $(LDFLAGS) $(LIBRARIES)

$(TARGET_8): $(DEPDIR) $(OBJECTS_8) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_8) $(LDFLAGS) $(LIBRARIES)

$(TARGET_9): $(DEPDIR) $(OBJECTS_9) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_9) $(LDFLAGS) $(LIBRARIES)

$(TARGET_10): $(DEPDIR) $(OBJECTS_10) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_10) $(LDFLAGS) $(LIBRARIES)

$(TARGET_11): $(DEPDIR) $(OBJECTS_11) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_11) $(LDFLAGS) $(LIBRARIES)

$(TARGET_12): $(DEPDIR) $(OBJECTS_12) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_12) $(LDFLAGS) $(LIBRARIES)

$(TARGET_13): $(DEPDIR) $(OBJECTS_13) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_13) $(LDFLAGS) $(LIBRARIES)

#$(TARGET_14): $(DEPDIR) $(OBJECTS_14) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so
#	$(LDSHARED) -o $@ $(OBJECTS_14) $(LDFLAGS) $(LIBRARIES)

$(TARGET_15): $(DEPDIR) $(OBJECTS_15) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_15) $(LDFLAGS) $(LIBRARIES)

$(TARGET_16): $(DEPDIR) $(OBJECTS_16) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_16) $(LDFLAGS) $(LIBRARIES)

$(TARGET_17): $(DEPDIR) $(OBJECTS_17) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_17) $(LDFLAGS) $(LIBRARIES)

$(TARGET_18): $(DEPDIR) $(OBJECTS_18) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_18) $(LDFLAGS) $(LIBRARIES)

$(TARGET_19): $(DEPDIR) $(OBJECTS_19) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so ../librave/scansun/libravescansun.so
	$(LDSHARED) -o $@ $(OBJECTS_19) $(LDFLAGS) -lravescansun $(LIBRARIES)

$(TARGET_20): $(DEPDIR) $(OBJECTS_20) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_20) $(LDFLAGS) $(LIBRARIES)

$(TARGET_21): $(DEPDIR) $(OBJECTS_21) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_21) $(LDFLAGS) $(LIBRARIES)

$(TARGET_22): $(DEPDIR) $(OBJECTS_22) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_22) $(LDFLAGS) $(LIBRARIES)

$(TARGET_23): $(DEPDIR) $(OBJECTS_23) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_23) $(LDFLAGS) $(LIBRARIES)

$(TARGET_24): $(DEPDIR) $(OBJECTS_24) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_24) $(LDFLAGS) $(LIBRARIES)

$(TARGET_25): $(DEPDIR) $(OBJECTS_25) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_25) $(LDFLAGS) $(LIBRARIES)

$(TARGET_26): $(DEPDIR) $(OBJECTS_26) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_26) $(LDFLAGS) $(LIBRARIES)

$(TARGET_27): $(DEPDIR) $(OBJECTS_27) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_27) $(LDFLAGS) $(LIBRARIES)

$(TARGET_28): $(DEPDIR) $(OBJECTS_28) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_28) $(LDFLAGS) $(LIBRARIES)

$(TARGET_29): $(DEPDIR) $(OBJECTS_29) ../librave/transform/libravetransform.so ../librave/pyapi/libravepyapi.so
	$(LDSHARED) -o $@ $(OBJECTS_29) $(LDFLAGS) $(LIBRARIES)

.PHONY=install
install:
	@mkdir -p ${prefix}/Lib
	@for i in $(TARGETS) ; \
	do \
		"$(HLHDF_INSTALL_BIN)" -f -o -m755 -C $$i "$(prefix)/Lib/$$i"; \
	done

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
#-include $(SOURCE_14:%.c=$(DEPDIR)/%.P)
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

