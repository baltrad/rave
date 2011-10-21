include ../def.mk

SCANSUNINC= -I. $(NUMPY_INCLUDE_DIR) $(INCLUDE_PYTHON) -I$(HLHDF_INCLUDE_DIR) \
	$(ZLIB_INCDIR) $(HDF5_INCDIR) $(PROJ_INCLUDE_DIR) \
	-I../transform

LDFLAGS+= -L. -L$(HLHDF_LIB_DIR) $(ZLIB_LIBDIR) $(HDF5_LIBDIR) $(PROJ_LIB_DIR) -L../transform

ifeq ($(EXPAT_SUPPRESSED), no)
SCANSUNINC+= $(EXPAT_INCLUDE_DIR)
LDFLAGS+= $(EXPAT_LIB_DIR)
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
SCANSUNLIBS= -lravescansun -lravetransform -lhlhdf -lproj  
ifeq ($(EXPAT_SUPPRESSED), no)
SCANSUNLIBS+= -lexpat
endif

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
	$(CC) $(SCANSUNINC) $(LDFLAGS) -o $(SCANSUNBIN) $(SCANSUNMAIN) $(SCANSUNLIBS)

.PHONY=install
install:
	@"$(HLHDF_INSTALL_BIN)" -f -o -C $(LIBRAVESCANSUN) "$(prefix)/lib/$(LIBRAVESCANSUN)"
	@for i in $(INSTALL_HEADERS) ; \
	do \
		"$(HLHDF_INSTALL_BIN)" -f -o -m644 -C $$i "$(prefix)/include/$$i"; \
	done
	"$(HLHDF_INSTALL_BIN)" -f -o -C $(SCANSUNBIN) "$(prefix)/bin/$(SCANSUNBIN)";

.PHONY=clean
clean:
		@\rm -f *.o core *~ $(SCANSUNBIN)
		@\rm -fr $(DEPDIR)

.PHONY=distclean		 
distclean:	clean
		@\rm -f *.so

# NOTE! This ensures that the dependencies are setup at the right time so this should not be moved
-include $(SCANSUNSOURCES:%.c=$(DEPDIR)/%.P)
