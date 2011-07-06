
include ../def.mk
include ../rules.mk
include sources.mk

DRAINUTILOBJS=	$(DRAINUTILSOURCES:util/%.cpp=%.o)

# All Target
all: $(DEPDIR) $(DRAINUTILOBJS)

# NOTE! This ensures that the dependencies are setup at the right time so this should not be moved
-include $(DRAINUTILSOURCES:util/%.cpp=$(DEPDIR)/%.P)

