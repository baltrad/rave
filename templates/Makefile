# Makefile template for RAVE - assumes that C code will be compiled 
# in a subdirectory to librave
# Change MYMODULE to the name of the module you are building.
# Do the same with some of the file names below under "Fixed definitions"
include ../def.mk

MYMODULEINC= -I. $(NUMPY_INCLUDE_DIR) $(INCLUDE_PYTHON) -I$(HLHDF_INCLUDE_DIR) \
	$(ZLIB_INCDIR) $(HDF5_INCDIR) $(PROJ_INCLUDE_DIR) \
	-I../transform

LDFLAGS+= -L. -L../transform -L/usr/local/lib -L$(HLHDF_LIB_DIR)

ifeq ($(EXPAT_SUPPRESSED), no)
MYMODULEINC+= $(EXPAT_INCLUDE_DIR)
LDFLAGS+= $(EXPAT_LIB_DIR)
endif

CFLAGS=	$(OPTS) $(CCSHARED) $(DEFS) $(CREATE_ITRUNC) $(MYMODULEINC)

# --------------------------------------------------------------------
# Fixed definitions

MYMODULESOURCES= mycode.c morecode.c  # <-- add your files here
INSTALL_HEADERS= mycode.h
MYMODULEOBJS=	$(MYMODULESOURCES:.c=.o)
LIBRAVEMYMODULE=	libravemymodule.so
MYMODULEMAIN= mycode_main.c
MYMODULEBIN= mybin
MYMODULELIBS= -lhlhdf -lproj -lravetransform -lravemymodule
ifeq ($(EXPAT_SUPPRESSED), no)
MYMODULELIBS+= -lexpat
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
all:		$(LIBRAVEMYMODULE) bin

$(LIBRAVEMYMODULE): $(DEPDIR) $(MYMODULEOBJS)
	$(LDSHARED) -o $@ $(MYMODULEOBJS)

.PHONY=bin
bin: 
	$(CC) $(MYMODULEINC) $(LDFLAGS) -o $(MYMODULEBIN) $(MYMODULEMAIN) $(MYMODULELIBS)

.PHONY=install
install:
	@"$(HLHDF_INSTALL_BIN)" -f -o -C $(LIBRAVEMYMODULE) "${DESTDIR}$(prefix)/lib/$(LIBRAVEMYMODULE)"
	@for i in $(INSTALL_HEADERS) ; \
	do \
		"$(HLHDF_INSTALL_BIN)" -f -o -m644 -C $$i "${DESTDIR}$(prefix)/include/$$i"; \
	done
	"$(HLHDF_INSTALL_BIN)" -f -o -C $(MYMODULEBIN) "${DESTDIR}$(prefix)/bin/$(MYMODULEBIN)";

.PHONY=clean
clean:
		@\rm -f *.o core *~ $(MYMODULEBIN)
		@\rm -fr $(DEPDIR)

.PHONY=distclean		 
distclean:	clean
		@\rm -f *.so

# NOTE! This ensures that the dependencies are setup at the right time so this should not be moved
-include $(MYMODULESOURCES:%.c=$(DEPDIR)/%.P)
