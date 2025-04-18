-include ../../def.mk

CFLAGS=	$(OPTS) $(CCSHARED) $(DEFS) $(CREATE_ITRUNC) \
	-I. -I../toolbox $(NUMPY_INCLUDE_DIR) $(INCLUDE_PYTHON) -I$(HLHDF_INCLUDE_DIR) $(PROJ_INCLUDE_DIR) \
	$(ZLIB_INCDIR) $(HDF5_INCDIR)

# --------------------------------------------------------------------
# Fixed definitions

RAVESOURCES=getpy.c h5rad.c rave.c pyravecompat.c pyraveapi.c
INSTALL_HEADERS= getpy.h h5rad.h rave.h pyraveapi.h
RAVEOBJS=	$(RAVESOURCES:.c=.o)
LIBRAVEPYAPI=	libravepyapi.so
MAKEDEPEND=gcc -MM $(CFLAGS) -o $(DF).d $<
DEPDIR=.dep
DF=$(DEPDIR)/$(*F)

# We reset TARGETS and INSTALL_HEADERS if we shouldn't compile for python. That way we don't need to destroy anything
ifeq ($(COMPILE_FOR_PYTHON), no)
LIBRAVEPYAPI=
INSTALL_HEADERS=
endif

# Get the dependencies in place, could be commented out if
# dependency generation not is supported by the compiler.

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
all: $(LIBRAVEPYAPI)

$(LIBRAVEPYAPI): $(DEPDIR) $(RAVEOBJS)
	$(LDSHARED) -o $@ $(RAVEOBJS)

ifeq ($(COMPILE_FOR_PYTHON), yes)
.PHONY=install
install:
	@"$(HLHDF_INSTALL_BIN)" -f -o -C $(LIBRAVEPYAPI) "${DESTDIR}$(prefix)/lib/$(LIBRAVEPYAPI)"
	@for i in $(INSTALL_HEADERS) ; \
	do \
		"$(HLHDF_INSTALL_BIN)" -f -o -m644 -C $$i "${DESTDIR}$(prefix)/include/$$i"; \
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
		@\rm -f *.so config.log config.status config.cache def.mk

# NOTE! This ensures that the dependencies are setup at the right time so this should not be moved
-include $(RAVESOURCES:%.c=$(DEPDIR)/%.P)
