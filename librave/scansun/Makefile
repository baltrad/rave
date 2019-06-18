-include ../../def.mk

SCANSUNINC= -I. $(NUMPY_INCLUDE_DIR) $(INCLUDE_PYTHON) -I$(HLHDF_INCLUDE_DIR) \
	$(ZLIB_INCDIR) $(HDF5_INCDIR) $(PROJ_INCLUDE_DIR) \
	-I../toolbox -I../tnc

LDFLAGS+= -L. -L$(HLHDF_LIB_DIR) $(ZLIB_LIBDIR) $(HDF5_LIBDIR) $(PROJ_LIB_DIR) -L../toolbox -L../tnc

ifeq ($(EXPAT_SUPPRESSED), no)
SCANSUNINC+= $(EXPAT_INCLUDE_DIR)
LDFLAGS+= $(EXPAT_LIB_DIR)
endif

ifeq ($(BUFR_SUPPRESSED), no)
SCANSUNINC+= $(BUFR_INCLUDE_DIR)
LDFLAGS+= $(BUFR_LIB_DIR)
endif

ifeq ($(NETCDF_SUPPRESSED), no)
SCANSUNINC+= $(NETCDF_INCLUDE_DIR)
LDFLAGS+= $(NETCDF_LIB_DIR)
endif

CFLAGS=	$(OPTS) $(CCSHARED) $(DEFS) $(CREATE_ITRUNC) $(SCANSUNINC)

# --------------------------------------------------------------------
# Fixed definitions

SCANSUNSOURCES= scansun.c
INSTALL_HEADERS= scansun.h
SCANSUNOBJS=	$(SCANSUNSOURCES:.c=.o)
LIBRAVESCANSUN=	libravescansun.so
SCANSUNMAIN= scansun_main.c
SCANSUNBIN= scansun
SCANSUNLIBS= -lravescansun -lravetoolbox -lhlhdf -lproj  -lravetnc
ifeq ($(BUFR_SUPPRESSED), no)
SCANSUNLIBS+= -lOperaBufr
endif 
ifeq ($(NETCDF_SUPPRESSED), no)
SCANSUNLIBS+= -lnetcdf
endif 
ifeq ($(EXPAT_SUPPRESSED), no)
SCANSUNLIBS+= -lexpat
endif

ifeq ($(GOT_PTHREAD_SUPPORT), yes)
SCANSUNLIBS+= -lpthread
endif

SCANSUNLIBS+= -lm

MAKEDEPEND=gcc -MM $(CFLAGS) -o $(DF).d $<
DEPDIR=.dep
DF=$(DEPDIR)/$(*F)
# --------------------------------------------------------------------
# Rules

# Contains dependency generation as well, so if you are not using
# gcc, comment out everything until the $(CC) statement.
%.o : %.c
	@$(MAKEDEPEND); \
	cp $(DF).d $(DF).P; \
	sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(DF).d >> $(DF).P; \
	\rm -f $(DF).d
	$(CC) -c $(CFLAGS) $<

# Ensures that the .dep directory exists
.PHONY=$(DEPDIR)
$(DEPDIR):
	+@[ -d $@ ] || mkdir -p $@

.PHONY=all
all:		$(LIBRAVESCANSUN) bin

$(LIBRAVESCANSUN): $(DEPDIR) $(SCANSUNOBJS)
	$(LDSHARED) -o $@ $(SCANSUNOBJS)

.PHONY=bin
bin: 
	$(CC) $(OPTS) $(DEFS) $(CREATE_ITRUNC) $(SCANSUNINC) $(LDFLAGS) -o $(SCANSUNBIN) $(SCANSUNMAIN) $(SCANSUNLIBS)

.PHONY=install
install:
	@"$(HLHDF_INSTALL_BIN)" -f -o -C $(LIBRAVESCANSUN) "${DESTDIR}$(prefix)/lib/$(LIBRAVESCANSUN)"
	@for i in $(INSTALL_HEADERS) ; \
	do \
		"$(HLHDF_INSTALL_BIN)" -f -o -m644 -C $$i "${DESTDIR}$(prefix)/include/$$i"; \
	done
	"$(HLHDF_INSTALL_BIN)" -f -o -C $(SCANSUNBIN) "${DESTDIR}$(prefix)/bin/$(SCANSUNBIN)";

.PHONY=clean
clean:
		@\rm -f *.o core *~ $(SCANSUNBIN)
		@\rm -fr $(DEPDIR)

.PHONY=distclean		 
distclean:	clean
		@\rm -f *.so

# NOTE! This ensures that the dependencies are setup at the right time so this should not be moved
-include $(SCANSUNSOURCES:%.c=$(DEPDIR)/%.P)
