
include ../def.mk
include ../rules.mk
include sources.mk

DRAINIMAGEOBJS=	$(DRAINIMAGESOURCES:image/%.cpp=%.o)

# All Target
all: $(DEPDIR) $(DRAINIMAGEOBJS)

# NOTE! This ensures that the dependencies are setup at the right time so this should not be moved
-include $(DRAINIMAGESOURCES:image/%.cpp=$(DEPDIR)/%.P)

