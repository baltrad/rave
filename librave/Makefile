# --------------------------------------------------------------------
# Fixed definitions
-include def.mk

# --------------------------------------------------------------------
# Rules

all: libtransform libpyapi libscansun

libtransform:
	$(MAKE) -C transform

libpyapi:
	$(MAKE) -C pyapi

libscansun:
	$(MAKE) -C scansun

def.mk: def.mk.in configure
	sh makelibs

# Remove rack specifics in the install step. This also removes the rack module
#
.PHONY=install
install:
	$(MAKE) -C transform install
	$(MAKE) -C pyapi install
	$(MAKE) -C scansun install
	@\rm -f ${prefix}/include/RackIf.h
	@\rm -f ${prefix}/lib/librack.*
	@\rm -f ${prefix}/lib/libdrain.*
	@\rm -fr ${prefix}/include/drain
	@\rm -f ${prefix}/Lib/_rack.so

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
	@\rm -f *.so config.log config.status config.cache def.mk
