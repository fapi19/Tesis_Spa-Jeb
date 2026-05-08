.PHONY: all thesis thesis-clean thesis-distclean

all: thesis

thesis:
	$(MAKE) -C thesis/latex pdf

thesis-clean:
	$(MAKE) -C thesis/latex clean

thesis-distclean:
	$(MAKE) -C thesis/latex distclean
