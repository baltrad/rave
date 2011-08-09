# Include ../def.mk first since some of its setting are overridden 
include ../def.mk
include def.mk
include main/sources.mk
include ropo/sources.mk
include hi5/sources.mk

#override some of the things set in ../def.mk
OPTS= -DNDEBUG -g -fwrapv -O2 -Wall

#Build default without support for ImageMagick
CCFLAGS+= -DDRAIN_MAGICK_no -DRACK_$RACK

RACKINC= -I. -I.. -I../drain/image -I../drain/radar -I../drain/util \
	-I../main -I../hi5 -I../ropo \
	-I./drain/image -I./drain/radar -I./drain/util \
	-I./main -I./hi5 -I./ropo \
	$(NUMPY_INCLUDE_DIR) $(INCLUDE_PYTHON) -I$(HLHDF_INCLUDE_DIR) \
	$(ZLIB_INCDIR) $(HDF5_INCDIR) $(PROJ_INCLUDE_DIR) \
	-I../../transform -I../transform

LDFLAGS+= -L. -L../transform -L../../transform -L$(HLHDF_LIB_DIR) $(ZLIB_LIBDIR) $(HDF5_LIBDIR) $(PROJ_LIB_DIR) -L./drain -L../drain 

ifeq ($(EXPAT_SUPPRESSED), no)
RACKINC+= $(EXPAT_INCLUDE_DIR)
LDFLAGS+= $(EXPAT_LIB_DIR)
endif

CFLAGS=	$(OPTS) $(CCSHARED) $(DEFS) $(CREATE_ITRUNC) $(RACKINC)

# --------------------------------------------------------------------
# Fixed definitions

INSTALL_HEADERS= main/RackIf.h
RACKOBJS=	$(RACKMAINSOURCES:.cpp=.o) $(RACKROPOSOURCES_C:.c=.o) $(RACKROPOSOURCES_CPP:.cpp=.o) $(RACKHI5SOURCES:.cpp=.o)
LIBRACK=	librack.so
RACKMAIN= main/main.cpp
RACKBIN= rack
RACKLIBS= -lhlhdf -lproj -lravetransform -lrack -ldrain -lpng
ifeq ($(EXPAT_SUPPRESSED), no)
RACKLIBS+= -lexpat
endif

DEPDIRS=main/$(DEPDIR) ropo/$(DEPDIR) hi5/$(DEPDIR)

# --------------------------------------------------------------------
# Rules

.PHONY=all
all:		$(LIBRACK) bin

buildmain:
	make -C main CFLAGS="${CFLAGS}" CCFLAGS="${CCFLAGS}" LDFLAGS="${LDFLAGS}" -k all

buildropo:
	make -C ropo -k all

buildhi5:
	make -C hi5 CFLAGS="${CFLAGS}" CCFLAGS="${CCFLAGS}" LDFLAGS="${LDFLAGS}" -k all

.PHONY=$(DEPDIRS)
$(DEPDIRS):
	mkdir -p $(DEPDIRS)

help:
	X="`cat Makefile Makefile-doxygen | grep '^[a-z\-]\+:' | cut -d' ' -f1 | tr ':' '|' `"; echo; echo make \[ $$X \]

$(LIBRACK): $(DEPDIRS) buildmain buildropo buildhi5 
	$(LDSHARED) -o $@ $(RACKOBJS)

.PHONY=bin
bin: 
	$(CC) $(RACKINC) $(LDFLAGS) -o $(RACKBIN) $(RACKMAIN) $(RACKLIBS)

.PHONY=install
install:
	@"$(HLHDF_INSTALL_BIN)" -f -o -C $(LIBRACK) "$(prefix)/lib/$(LIBRACK)"
	@for i in $(INSTALL_HEADERS) ; \
	do \
		cp -v $$i "$(prefix)/include/"; \
	done  
	"$(HLHDF_INSTALL_BIN)" -f -o -C $(RACKBIN) "$(prefix)/bin/$(RACKBIN)";

.PHONY=clean
clean:
		@\rm -f main/*.o ropo/*.o hi5/*.o core *~ $(RACKBIN)
		@\rm -fr $(DEPDIRS)

.PHONY=distclean		 
distclean:	clean
		@\rm -f *.so

# DOCUMENTATION
dox-all: help.txt help-drain.txt help-andre.txt examples.txt dox
	touch help.txt

examples.txt:
	grep '^[ ]\+FILES\?=.*'  main/*.dox > $@
	grep '^[ ]\+rack [^<]*$$' main/*.dox >> $@

help.txt:
	./rack --help > $@

help-drain.txt:
	./rack --man drain > $@

help-andre.txt:
	./rack --man andre > $@
	./rack --man ropo >> $@

include Makefile-doxygen
