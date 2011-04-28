
include ../def.mk
include ../rules.mk
include sources.mk

RACKMAINOBJS=	$(RACKMAINSOURCES:main/%.cpp=%.o)

# All Target
all: $(DEPDIR) $(RACKMAINOBJS)

# NOTE! This ensures that the dependencies are setup at the right time so this should not be moved
-include $(RACKMAINSOURCES:main/%.cpp=$(DEPDIR)/%.P)

