
include ../../def.mk
include ../def.mk
#include ../rules.mk
include sources.mk

CC=		gcc
CFLAGS+=-Wall -pedantic -I../../transform -I. 
CPPFLAGS+=-I../
LDFLAGS+=-L$(HLHDF_LIB_DIR) -L../../transform -L.. -L. -lm -lproj -lhlhdf -lstdc++ -lravetransform -lpng
LDSHARED=	$(CC) -shared
CCSHARED=	-fPIC

SOURCES=$(RACKROPOSOURCES_C:ropo/%.c=%.c)
OBJS=$(SOURCES:%.c=%.o)
EXEC=ropo
LIB=libropo.so

# Contains dependency generation as well, so if you are not using
# gcc, comment out everything until the $(CC) statement.
%.o: %.c
	@$(MAKEDEPEND); \
	cp $(DF).d $(DF).P; \
	sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(DF).d >> $(DF).P; \
	\rm -f $(DF).d
	$(CC) -c $(CCSHARED) $(CFLAGS) $(CCFLAGS) $<

%.o: %.cpp
	@$(MAKEDEPEND); \
	cp $(DF).d $(DF).P; \
	sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(DF).d >> $(DF).P; \
	\rm -f $(DF).d
	$(CC) -c $(CCSHARED) $(CPPFLAGS) $(CFLAGS) $(CCFLAGS) $<

$(EXEC): $(LIB) main.c main_help.c
	$(CC) $(CFLAGS) $(LDFLAGS) main.c -o $@ -lropo

$(LIB): $(OBJS)
	$(LDSHARED) $(CCSHARED) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJS)

.PHONY=clean install

clean:
	rm -f *~ $(EXEC) $(LIB) $(OBJS)

install:
	@"$(HLHDF_INSTALL_BIN)" -f -o -C $(LIB) "$(prefix)/lib/$(LIB)"
	"$(HLHDF_INSTALL_BIN)" -f -o -C $(EXEC) "$(prefix)/bin/$(EXEC)";

# Ensures that the .dep directory exists
.PHONY=$(DEPDIR)
$(DEPDIR):
	+@[ -d $@ ] || mkdir -p $@

RACKROPOOBJS_C=	$(RACKROPOSOURCES_C:ropo/%.c=%.o)
RACKROPOOBJS_CPP= $(RACKROPOSOURCES_CPP:ropo/%.cpp=%.o)

# All Target
all: $(DEPDIR) $(RACKROPOOBJS_C) $(RACKROPOOBJS_CPP) $(EXEC)

# NOTE! This ensures that the dependencies are setup at the right time so this should not be moved
-include $(RACKROPOSOURCES_C:ropo/%.c=$(DEPDIR)/%.P)
-include $(RACKROPOSOURCES_CPP:ropo/%.cpp=$(DEPDIR)/%.P)

