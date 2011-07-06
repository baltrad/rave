
include ../def.mk
include ../rules.mk
include sources.mk

RACKHI5OBJS=	$(RACKHI5SOURCES:hi5/%.cpp=%.o)

# All Target
all: $(DEPDIR) $(RACKHI5OBJS)

# NOTE! This ensures that the dependencies are setup at the right time so this should not be moved
-include $(RACKHI5SOURCES:hi5/%.cpp=$(DEPDIR)/%.P)

