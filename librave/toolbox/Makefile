-include ../../def.mk

CFLAGS=	$(OPTS) $(CCSHARED) $(DEFS) $(CREATE_ITRUNC) \
	-I. $(NUMPY_INCLUDE_DIR) $(INCLUDE_PYTHON) -I$(HLHDF_INCLUDE_DIR) $(PROJ_INCLUDE_DIR) \
	$(ZLIB_INCDIR) $(HDF5_INCDIR)

ifeq ($(EXPAT_SUPPRESSED), no)
CFLAGS+= $(EXPAT_INCLUDE_DIR) -DRAVE_XML_SUPPORTED
endif

ifeq ($(BUFR_SUPPRESSED), no)
CFLAGS+= $(BUFR_INCLUDE_DIR) -DRAVE_BUFR_SUPPORTED
endif

ifeq ($(NETCDF_SUPPRESSED),no)
CFLAGS+= $(NETCDF_INCLUDE_DIR)
ifeq ($(CF_SUPPRESSED), no)
CFLAGS+= -DRAVE_CF_SUPPORTED
endif
endif

ifeq ($(GOT_PTHREAD_SUPPORT), yes)
CFLAGS+= -DPTHREAD_SUPPORTED
endif

# --------------------------------------------------------------------
# Fixed definitions

RAVESOURCES= polar.c raveutil.c rave_transform.c rave_alloc.c rave_debug.c polarvolume.c \
             polarscan.c polarscanparam.c cartesian.c cartesianparam.c cartesianvolume.c transform.c projection.c projection_pipeline.c polarnav.c \
             rave_io.c rave_list.c rave_object.c raveobject_list.c area.c rave_datetime.c \
             rave_types.c rave_data2d.c composite.c rave_attribute.c rave_attribute_table.c cartesiancomposite.c \
             rave_utilities.c rave_field.c radardefinition.c rave_hlhdf_utilities.c cartesian_odim_io.c \
             polar_odim_io.c raveobject_hashtable.c detection_range.c odim_io_utilities.c poo_composite_algorithm.c rave_acrr.c \
             vertical_profile.c vp_odim_io.c dealias.c odc_hac.c rave_qitotal.c rave_gra.c ctfilter.c bitmap_generator.c \
             proj_wkt_helper.c lazy_nodelist_reader.c lazy_dataset.c acqva.c rave_iocache.c rave_legend.c compositegenerator.c \
						 compositearguments.c legacycompositegeneratorfactory.c acqvacompositegeneratorfactory.c odim_source.c \
						 composite_utils.c compositefilter.c compositefactorymanager.c compositeengine.c ratecompositegeneratorfactory.c

ifeq ($(EXPAT_SUPPRESSED), no)
RAVESOURCES += arearegistry.c projectionregistry.c rave_simplexml.c odim_sources.c 
endif

ifeq ($(BUFR_SUPPRESSED), no)
RAVESOURCES += rave_bufr_io.c
endif 

ifeq ($(CF_SUPPRESSED),no)
RAVESOURCES += cartesian_cf_io.c
endif

INSTALL_HEADERS= polar.h rave_transform.h raveutil.h rave_alloc.h polarvolume.h polarscan.h \
                 polarscanparam.h cartesian.h cartesianparam.h cartesianvolume.h transform.h projection.h projection_pipeline.h polarnav.h rave_io.h \
                 rave_list.h rave_object.h raveobject_list.h area.h rave_datetime.h \
                 rave_types.h rave_data2d.h composite.h rave_attribute.h rave_attribute_table.h cartesiancomposite.h \
                 rave_utilities.h rave_field.h radardefinition.h rave_hlhdf_utilities.h \
                 cartesian_odim_io.h rave_debug.h polar_odim_io.h \
                 raveobject_hashtable.h detection_range.h odim_io_utilities.h composite_algorithm.h poo_composite_algorithm.h rave_acrr.h \
                 vertical_profile.h vp_odim_io.h dealias.h odc_hac.h  rave_qitotal.h rave_gra.h ctfilter.h  bitmap_generator.h \
                 proj_wkt_helper.h lazy_nodelist_reader.h lazy_dataset.h rave_proj.h acqva.h rave_iocache.h rave_legend.h \
				 				 compositegenerator.h compositearguments.h compositegeneratorfactory.h legacycompositegeneratorfactory.h \
								 acqvacompositegeneratorfactory.h odim_source.h composite_utils.h compositefilter.h compositefactorymanager.h compositeengine.h  ratecompositegeneratorfactory.h

ifeq ($(EXPAT_SUPPRESSED), no)
INSTALL_HEADERS+= arearegistry.h projectionregistry.h rave_simplexml.h odim_sources.h 
endif

ifeq ($(BUFR_SUPPRESSED), no)
INSTALL_HEADERS += rave_bufr_io.h
endif 

ifeq ($(CF_SUPPRESSED),no)
INSTALL_HEADERS += cartesian_cf_io.h
endif

RAVEOBJS=	$(RAVESOURCES:.c=.o)
LIBRAVETOOLBOX=	libravetoolbox.so

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
all:		$(LIBRAVETOOLBOX)

$(LIBRAVETOOLBOX): $(DEPDIR) $(RAVEOBJS)
	$(LDSHARED) -o $@ $(RAVEOBJS)

.PHONY=install
install:
	@"$(HLHDF_INSTALL_BIN)" -f -o -C $(LIBRAVETOOLBOX) "${DESTDIR}$(prefix)/lib/$(LIBRAVETOOLBOX)"
	@for i in $(INSTALL_HEADERS) ; \
	do \
		"$(HLHDF_INSTALL_BIN)" -f -o -m644 -C $$i "${DESTDIR}$(prefix)/include/$$i"; \
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
