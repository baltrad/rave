include ../def.mk

CFLAGS=	$(OPTS) $(CCSHARED) $(DEFS) $(CREATE_ITRUNC) \
	-I. $(NUMPY_INCLUDE_DIR) $(INCLUDE_PYTHON) -I$(HLHDF_INCLUDE_DIR) \
	$(ZLIB_INCDIR) $(HDF5_INCDIR) $(PROJ_INCLUDE_DIR)

# --------------------------------------------------------------------
# Fixed definitions

RAVESOURCES= polar.c raveutil.c rave_transform.c rave_alloc.c rave_debug.c polarvolume.c \
             polarscan.c polarscanparam.c cartesian.c cartesianvolume.c transform.c projection.c polarnav.c \
             rave_io.c rave_list.c rave_object.c raveobject_list.c area.c rave_datetime.c \
             rave_types.c rave_data2d.c composite.c raveobject_hashtable.c rave_attribute.c \
             rave_utilities.c rave_field.c radardefinition.c rave_hlhdf_utilities.c cartesian_odim_io.c \
             polar_odim_io.c
             
INSTALL_HEADERS= polar.h rave_transform.h raveutil.h rave_alloc.h polarvolume.h polarscan.h \
                 polarscanparam.h cartesian.h cartesianvolume.h transform.h projection.h polarnav.h rave_io.h \
                 rave_list.h rave_object.h raveobject_list.h area.h rave_datetime.h \
                 rave_types.h rave_data2d.h composite.h raveobject_hashtable.h rave_attribute.h \
                 rave_utilities.h rave_field.h radardefinition.h rave_hlhdf_utilities.h \
                 cartesian_odim_io.h rave_debug.h polar_odim_io.h

RAVEOBJS=	$(RAVESOURCES:.c=.o)
LIBRAVETRANSFORM=	libravetransform.so

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
all:		$(LIBRAVETRANSFORM)

$(LIBRAVETRANSFORM): $(DEPDIR) $(RAVEOBJS)
	$(LDSHARED) -o $@ $(RAVEOBJS)

.PHONY=install
install:
	@"$(HLHDF_INSTALL_BIN)" -f -o -C $(LIBRAVETRANSFORM) "$(prefix)/lib/$(LIBRAVETRANSFORM)"
	@for i in $(INSTALL_HEADERS) ; \
	do \
		"$(HLHDF_INSTALL_BIN)" -f -o -m644 -C $$i "$(prefix)/include/$$i"; \
	done

.PHONY=clean
clean:
		@\rm -f *.o core *~
		@\rm -fr $(DEPDIR)

.PHONY=distclean		 
distclean:	clean
		@\rm -f *.so config.log config.status config.cache def.mk

# NOTE! This ensures that the dependencies are setup at the right time so this should not be moved
-include $(RAVESOURCES:%.c=$(DEPDIR)/%.P)
