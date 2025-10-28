# Makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help clean html livehtml linkcheck coverage strict

# Clean build artifacts
clean:
	@echo "Cleaning build directory..."
	@rm -rf $(BUILDDIR)
	@echo "Clean complete."

# Build HTML documentation
html:
	@echo "Building HTML documentation..."
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

# Build HTML with live reload (requires sphinx-autobuild)
livehtml:
	@echo "Starting live documentation server..."
	@sphinx-autobuild \
		--open-browser \
		--delay 1 \
		--ignore "*.swp" \
		--ignore "*.tmp" \
		--ignore "*~" \
		"$(SOURCEDIR)" "$(BUILDDIR)/html" \
		$(SPHINXOPTS) $(O)

# Check documentation links
linkcheck:
	@echo "Checking documentation links..."
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "Link check complete. See $(BUILDDIR)/linkcheck/output.txt"

# Check documentation coverage
coverage:
	@echo "Checking documentation coverage..."
	@$(SPHINXBUILD) -b coverage "$(SOURCEDIR)" "$(BUILDDIR)/coverage" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "Coverage check complete. See $(BUILDDIR)/coverage/python.txt"

# Build with warnings as errors (strict mode)
strict:
	@echo "Building documentation in strict mode (warnings as errors)..."
	@$(SPHINXBUILD) -b html -W --keep-going "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
