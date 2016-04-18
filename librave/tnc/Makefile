-include ../../def.mk

CFLAGS=	$(OPTS) $(CCSHARED) $(DEFS)

ifeq ($(GOT_PTHREAD_SUPPORT), yes)
CFLAGS+= -DPTHREAD_SUPPORTED
endif

# --------------------------------------------------------------------
# Fixed definitions

SOURCES= tnc.c

INSTALL_HEADERS= tnc.h 

OBJECTS=	$(SOURCES:.c=.o)
LIBTNC=	libravetnc.so

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
all:		$(LIBTNC)

$(LIBTNC): $(DEPDIR) $(OBJECTS)
	$(LDSHARED) -o $@ $(OBJECTS)

.PHONY=install
install:
	@"$(HLHDF_INSTALL_BIN)" -f -o -C $(OBJECTS) "$(prefix)/lib/$(LIBTNC)"
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
-include $(SOURCES:%.c=$(DEPDIR)/%.P)

