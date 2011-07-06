# --------------------------------------------------------------------
# Fixed definitions

# --------------------------------------------------------------------
# Rules

all: libtransform libpyapi libscansun libdrain librack

libtransform:
	$(MAKE) -C transform

libpyapi:
	$(MAKE) -C pyapi

libscansun:
	$(MAKE) -C scansun

libdrain:
	$(MAKE) -C rack/drain

librack:
	$(MAKE) -C rack

def.mk: def.mk.in configure
	sh makelibs

.PHONY=install
install:
	$(MAKE) -C transform install
	$(MAKE) -C pyapi install
	$(MAKE) -C scansun install
	$(MAKE) -C rack/drain install
	$(MAKE) -C rack install

.PHONY=clean
clean:
	-$(MAKE) -C transform clean
	-$(MAKE) -C pyapi clean
	-$(MAKE) -C scansun clean
	-$(MAKE) -C rack/drain clean
	-$(MAKE) -C rack clean
	@\rm -f *.o core *~

.PHONY=distclean
distclean:	clean
	-$(MAKE) -C transform distclean
	-$(MAKE) -C pyapi distclean
	-$(MAKE) -C scansun distclean
	-$(MAKE) -C rack/drain distclean
	-$(MAKE) -C rack distclean
	@\rm -f *.so config.log config.status config.cache def.mk
