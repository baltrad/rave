###########################################################################
# Copyright (C) 2011 Swedish Meteorological and Hydrological Institute, SMHI,
#
# This file is part of Radvol-QC.
#
# Radvol-QC is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Radvol-QC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with RAVE.  If not, see <http://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------
# 
# Radvol-QC make file
# @file
# @author
# @date
###########################################################################
-include ../../../def.mk

# C flags, use RAVE suggested ones: overkill ... but it works
#
CFLAGS= -I. $(CCSHARED) $(OPTS) $(NUMPY_INCLUDE_DIR) $(INCLUDE_PYTHON) \
	-I$(HLHDF_INCLUDE_DIR) $(ZLIB_INCDIR) $(HDF5_INCDIR) \
	$(PROJ_INCLUDE_DIR) -I../../toolbox

LDFLAGS+= -L. -L$(HLHDF_LIB_DIR) $(ZLIB_LIBDIR) $(HDF5_LIBDIR) \
	  $(PROJ_LIB_DIR) -L../../toolbox

# --------------------------------------------------------------------
# Fixed definitions

SOURCES= radvol.c radvolatt.c radvolbroad.c radvolspeck.c radvolspike.c 

OBJECTS= $(SOURCES:.c=.o)

TARGET= libradvol.so

MAKECDEPEND=$(CC) -MM $(CFLAGS) -MT '$(@F)' -o  $(DF).d $<

DEPDIR=.dep
DF=$(DEPDIR)/$(*F)

# Ensures that the .dep directory exists
.PHONY=$(DEPDIR)
$(DEPDIR):
	+@[ -d $@ ] || mkdir -p $@

# And the rest of the make file targets
#
.PHONY=all
all:		$(TARGET)

$(TARGET): $(DEPDIR) $(OBJECTS)
	$(LDSHARED) $(LDFLAGS) -o $@ $(OBJECTS)

.PHONY=install
install:
	@mkdir -p ${prefix}/include/
	@mkdir -p ${prefix}/lib/
	@cp -v -f *.h ${prefix}/include/
	@cp -v -f $(TARGET) ${prefix}/lib/

.PHONY=clean
clean:
	@\rm -f *.o core *~
	@\rm -fr $(DEPDIR)

.PHONY=distclean		 
distclean:	clean
	@\rm -f $(TARGET)

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
-include $(SOURCES:%.c=$(DEPDIR)/%.P)
