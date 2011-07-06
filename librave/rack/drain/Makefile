# Include ../../def.mk first since some of its setting are overridden 
include ../../def.mk
include def.mk
include image/sources.mk
include util/sources.mk
include radar/sources.mk

#override some of the things set in ../def.mk
OPTS= -DNDEBUG -g -fwrapv -O2 -Wall

#Build default without support for ImageMagick
CCFLAGS+= -DDRAIN_MAGICK_no

DRAININC= -I. -I./image -I./radar -I./util \
	$(NUMPY_INCLUDE_DIR) $(INCLUDE_PYTHON) -I$(HLHDF_INCLUDE_DIR) \
	$(ZLIB_INCDIR) $(HDF5_INCDIR) $(PROJ_INCLUDE_DIR)

LDFLAGS+= -L. -L/usr/local/lib -L/usr/lib -L$(HLHDF_LIB_DIR)

ifeq ($(EXPAT_SUPPRESSED), no)
DRAININC+= $(EXPAT_INCLUDE_DIR)
LDFLAGS+= $(EXPAT_LIB_DIR)
endif

CFLAGS=	$(OPTS) $(CCSHARED) $(DEFS) $(CREATE_ITRUNC) $(DRAININC)

# --------------------------------------------------------------------
# Fixed definitions

INSTALL_HEADERS= #Should be all header files in image, radar och util
DRAINOBJS=	$(DRAINIMAGESOURCES:.cpp=.o) $(DRAINRADARSOURCES:.cpp=.o) $(DRAINUTILSOURCES:.cpp=.o)
LIBDRAIN=	libdrain.so
DEPDIRS=image/$(DEPDIR) radar/$(DEPDIR) util/$(DEPDIR)
SUBDIRS= image radar util

# --------------------------------------------------------------------
# Rules

.PHONY=all
all:		$(LIBDRAIN)
	@echo 'Building drain'

buildimage:
	make -C image CFLAGS="${CFLAGS}" CCFLAGS="${CCFLAGS}" LDFLAGS="${LDFLAGS}" -k all

buildradar:
	make -C radar CFLAGS="${CFLAGS}" CCFLAGS="${CCFLAGS}" LDFLAGS="${LDFLAGS}" -k all

buildutil:
	make -C util CFLAGS="${CFLAGS}" CCFLAGS="${CCFLAGS}" LDFLAGS="${LDFLAGS}" -k all

.PHONY=$(DEPDIRS)
$(DEPDIRS):
	mkdir -p $(DEPDIRS)

help:
	X="`cat Makefile Makefile-doxygen | grep '^[a-z\-]\+:' | cut -d' ' -f1 | tr ':' '|' `"; echo; echo make \[ $$X \]

$(LIBDRAIN): $(DEPDIRS) buildimage buildradar buildutil
	$(LDSHARED) -o $@ $(DRAINOBJS)

.PHONY=install
install:
	@echo 'copying drainlib to $(prefix)/lib/$(LIBDRAIN)' 	
	@"$(HLHDF_INSTALL_BIN)" -f -o -C $(LIBDRAIN) "$(prefix)/lib/$(LIBDRAIN)"
	@for i in $(SUBDIRS) ; \
	do \
		mkdir -p "$(prefix)/include/drain/$$i"; \
		cp -v $$i/*.h "$(prefix)/include/drain/$$i"; \
	done

.PHONY=clean
clean:
	@\rm -f image/*.o radar/*.o util/*.o *~		
	@\rm -fr $(DEPDIRS)

.PHONY=distclean		 
distclean:	clean
		@\rm -f *.so

