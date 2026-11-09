# UGM Thesis LaTeX Template - Makefile
# Supports local compilation (requires texlive) and Docker compilation

.PHONY: help thesis skripsi clean thesis-docker skripsi-docker shell all all-docker

# Default target
help:
	@echo "UGM Thesis LaTeX Template - Build Options"
	@echo "========================================="
	@echo ""
	@echo "Local Compilation (requires texlive):"
	@echo "  make thesis            - Compile template/thesis/main.tex"
	@echo "  make clean             - Remove build artifacts"
	@echo ""
	@echo "Docker Compilation (no texlive required):"
	@echo "  make thesis-docker     - Compile thesis in Docker container"
	@echo "  make shell             - Open interactive shell in container"
	@echo ""
	@echo "CI/All:"
	@echo "  make all               - Compile thesis"
	@echo "  make all-docker        - Compile thesis in Docker"

# Local compilation
thesis:
	cd template/thesis && \
		pdflatex -interaction=nonstopmode main.tex && \
		bibtex main && \
		pdflatex -interaction=nonstopmode main.tex && \
		pdflatex -interaction=nonstopmode main.tex

skripsi: thesis

clean:
	find template -name "*.aux" -o -name "*.log" -o -name "*.out" -o -name "*.toc" -o -name "*.lof" -o -name "*.lot" -o -name "*.fls" -o -name "*.fdb_latexmk" -o -name "*.synctex.gz" -o -name "*.bbl" -o -name "*.blg" -o -name "*.glo" | xargs rm -f 2>/dev/null || true

# Docker compilation
DOCKER_IMAGE := ugm-thesis-latex
DOCKER_TAG := latest

build-docker:
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) -f docker/Dockerfile .

thesis-docker: build-docker
	docker run --rm -v $(PWD)/template:/workspace $(DOCKER_IMAGE):$(DOCKER_TAG) sh -c "cd /workspace/thesis && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex"

skripsi-docker: thesis-docker

shell: build-docker
	docker run --rm -it -v $(PWD)/template:/workspace $(DOCKER_IMAGE):$(DOCKER_TAG) sh

# Compile all
all: thesis

all-docker: thesis-docker
