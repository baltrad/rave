# --------------------------------------------------------------------
# Fixed definitions

# --------------------------------------------------------------------
# Rules

all: libtransform libpyapi

libtransform:
	$(MAKE) -C transform

libpyapi:
	$(MAKE) -C pyapi

def.mk: def.mk.in configure
	sh makelibs

.PHONY=install
install:
	$(MAKE) -C transform install
	$(MAKE) -C pyapi install

.PHONY=clean
clean:
	-$(MAKE) -C transform clean
	-$(MAKE) -C pyapi clean
	@\rm -f *.o core *~

.PHONY=distclean
distclean:	clean
	-$(MAKE) -C transform distclean
	-$(MAKE) -C pyapi distclean
	@\rm -f *.so config.log config.status config.cache def.mk
