# Minimal Makefile for Sphinx documentation
# Run from repo root:  cd docs && make html

SPHINXBUILD = sphinx-build
SOURCEDIR   = .
BUILDDIR    = _build

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)
	@echo "Open $(BUILDDIR)/index.html in a browser."

clean:
	rm -rf $(BUILDDIR)
