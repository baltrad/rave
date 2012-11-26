# --------------------------------------------------------------------
# Fixed definitions

# --------------------------------------------------------------------
# Rules

all: libtoolbox libpyapi libscansun libradvol

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
	$(MAKE) -C toolbox install
	$(MAKE) -C pyapi install
	$(MAKE) -C scansun install
	$(MAKE) -C radvol install

.PHONY=clean
clean:
	-$(MAKE) -C toolbox clean
	-$(MAKE) -C pyapi clean
	-$(MAKE) -C scansun clean
	-$(MAKE) -C radvol clean
	@\rm -f *.o core *~

.PHONY=distclean
distclean:	clean
	-$(MAKE) -C toolbox distclean
	-$(MAKE) -C pyapi distclean
	-$(MAKE) -C scansun distclean
	-$(MAKE) -C radvol distclean
	