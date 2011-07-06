
include ../def.mk
#include ../rules.mk
include sources.mk


# Contains dependency generation as well, so if you are not using
# gcc, comment out everything until the $(CC) statement.
%.o: %.c
	@$(MAKEDEPEND); \
	cp $(DF).d $(DF).P; \
	sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(DF).d >> $(DF).P; \
	\rm -f $(DF).d
	gcc -c $(CFLAGS) $(CCFLAGS) $<

%.o: %.cpp
	@$(MAKEDEPEND); \
	cp $(DF).d $(DF).P; \
	sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(DF).d >> $(DF).P; \
	\rm -f $(DF).d
	$(CC) -c $(CFLAGS) $(CCFLAGS) $<


# Ensures that the .dep directory exists
.PHONY=$(DEPDIR)
$(DEPDIR):
	+@[ -d $@ ] || mkdir -p $@

RACKROPOOBJS_C=	$(RACKROPOSOURCES_C:ropo/%.c=%.o)
RACKROPOOBJS_CPP= $(RACKROPOSOURCES_CPP:ropo/%.cpp=%.o)

# All Target
all: $(DEPDIR) $(RACKROPOOBJS_C) $(RACKROPOOBJS_CPP)

# NOTE! This ensures that the dependencies are setup at the right time so this should not be moved
-include $(RACKROPOSOURCES_C:ropo/%.c=$(DEPDIR)/%.P)
-include $(RACKROPOSOURCES_CPP:ropo/%.cpp=$(DEPDIR)/%.P)

