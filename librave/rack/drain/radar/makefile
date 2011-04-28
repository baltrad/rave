
include ../def.mk
include ../rules.mk
include sources.mk

DRAINRADAROBJS=	$(DRAINRADARSOURCES:radar/%.cpp=%.o)

# All Target
all: $(DEPDIR) $(DRAINRADAROBJS)

# NOTE! This ensures that the dependencies are setup at the right time so this should not be moved
-include $(DRAINRADARSOURCES:radar/%.cpp=$(DEPDIR)/%.P)

