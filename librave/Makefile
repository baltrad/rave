# --------------------------------------------------------------------
# Fixed definitions

# --------------------------------------------------------------------
# Rules

all: libtransform libpyapi libscansun

libtransform:
	$(MAKE) -C transform

libpyapi:
	$(MAKE) -C pyapi

libscansun:
	$(MAKE) -C scansun

.PHONY=install
install:
	$(MAKE) -C transform install
	$(MAKE) -C pyapi install
	$(MAKE) -C scansun install

.PHONY=clean
clean:
	-$(MAKE) -C transform clean
	-$(MAKE) -C pyapi clean
	-$(MAKE) -C scansun clean
	@\rm -f *.o core *~

.PHONY=distclean
distclean:	clean
	-$(MAKE) -C transform distclean
	-$(MAKE) -C pyapi distclean
	-$(MAKE) -C scansun distclean
