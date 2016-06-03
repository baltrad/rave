include ../def.mk

CFLAGS=	$(OPTS) $(CCSHARED) $(DEFS) $(CREATE_ITRUNC) \
	-I. -I../librave/tnc -I../librave/toolbox $(RAVE_MODULE_CFLAGS)

LDFLAGS= $(RAVE_MODULE_LDFLAGS)
# --------------------------------------------------------------------
# Fixed definitions

AREAREGISTRY_SOURCES= arearegistry_access.c

AREAREGISTRY_OBJECTS=	$(AREAREGISTRY_SOURCES:.c=.o)

AREAREGISTRY_BIN=arearegistry

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
all:		$(AREAREGISTRY_BIN)

$(AREAREGISTRY_BIN): $(DEPDIR) $(AREAREGISTRY_OBJECTS) ../librave/toolbox/libravetoolbox.so
	$(CC) $(LDFLAGS) -o $@ $(AREAREGISTRY_OBJECTS) -L../librave/tnc -L../librave/toolbox $(RAVE_MODULE_LIBRARIES) -lm

.PHONY=install
install: ;

.PHONY=clean
clean:
		@\rm -f *.o core *~
		@\rm -fr $(DEPDIR)

.PHONY=distclean		 
distclean:	clean
		@\rm -f $(AREAREGISTRY_BIN)

# NOTE! This ensures that the dependencies are setup at the right time so this should not be moved
-include $(AREAREGISTRY_SOURCES:%.c=$(DEPDIR)/%.P)

