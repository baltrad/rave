-include ../def.mk

# --------------------------------------------------------------------
# Fixed definitions

# --------------------------------------------------------------------
# Rules
TARGETS=libtnc libtoolbox libpyapi libscansun

ifeq ($(EXPAT_SUPPRESSED), no)
TARGETS += libradvol
endif

.PHONY=all
all: $(TARGETS)

libtnc:
	$(MAKE) -C tnc

libtoolbox:
	$(MAKE) -C toolbox

libpyapi:
	$(MAKE) -C pyapi

libscansun:
	$(MAKE) -C scansun
	
libradvol:
	$(MAKE) -C radvol

.PHONY=install
install:
	$(MAKE) -C tnc install
	$(MAKE) -C toolbox install
	$(MAKE) -C pyapi install
	$(MAKE) -C scansun install
ifeq ($(EXPAT_SUPPRESSED), no)
	$(MAKE) -C radvol install
endif

.PHONY=clean
clean:
	-$(MAKE) -C tnc clean
	-$(MAKE) -C toolbox clean
	-$(MAKE) -C pyapi clean
	-$(MAKE) -C scansun clean
	-$(MAKE) -C radvol clean
	@\rm -f *.o core *~

.PHONY=distclean
distclean:	clean
	-$(MAKE) -C tnc distclean
	-$(MAKE) -C toolbox distclean
	-$(MAKE) -C pyapi distclean
	-$(MAKE) -C scansun distclean
	-$(MAKE) -C radvol distclean
	