
#override compiler set by ../def.mk
CC=g++

#Set dependencies
MAKEDEPEND=g++ -MM $(CFLAGS) $(CCFLAGS) -o $(DF).d $<
DEPDIR=.dep
DF=$(DEPDIR)/$(*F)

