# --------------------------------------------------------------------
# Fixed definitions

# --------------------------------------------------------------------
# Rules

all: libtoolbox libpyapi libscansun

libtoolbox:
	$(MAKE) -C toolbox

libpyapi:
	$(MAKE) -C pyapi

libscansun:
	$(MAKE) -C scansun

.PHONY=install
install:
	$(MAKE) -C toolbox install
	$(MAKE) -C pyapi install
	$(MAKE) -C scansun install

.PHONY=clean
clean:
	-$(MAKE) -C toolbox clean
	-$(MAKE) -C pyapi clean
	-$(MAKE) -C scansun clean
	@\rm -f *.o core *~

.PHONY=distclean
distclean:	clean
	-$(MAKE) -C toolbox distclean
	-$(MAKE) -C pyapi distclean
	-$(MAKE) -C scansun distclean
