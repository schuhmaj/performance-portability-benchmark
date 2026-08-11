# Renders the figures in this directory: <name>.tex -> <name>.pdf -> <name>.png
#
#   make            build the PDF and the PNG of every *.tex here
#   make pdf        stop after the PDF
#   make DENSITY=400 png    rasterize at a higher resolution
#   make clean      drop the LaTeX auxiliary files
#   make distclean  also drop the generated PDFs and PNGs
#
# Requires a TeX distribution with latexmk, and ImageMagick with its Ghostscript
# delegate for the PDF -> PNG step.

# Rasterization resolution in DPI. -density is passed *before* the input file so
# that Ghostscript renders at this resolution instead of upscaling a 72 DPI page.
DENSITY ?= 200

LATEXMK ?= latexmk
MAGICK  ?= magick

LATEXMKFLAGS ?= -pdf -silent -halt-on-error
# Flatten the transparent PDF background onto white, otherwise the dark text of
# the figure is unreadable for anyone viewing GitHub in dark mode.
MAGICKFLAGS  ?= -background white -alpha remove -alpha off

SOURCES := $(wildcard *.tex)
PDFS    := $(SOURCES:.tex=.pdf)
PNGS    := $(SOURCES:.tex=.png)

.PHONY: all pdf png clean distclean

all: $(PNGS)

pdf: $(PDFS)

png: $(PNGS)

# latexmk does its own dependency tracking and leaves the PDF untouched when
# nothing changed. Without the trailing touch its timestamp would stay behind
# the .tex and make would re-run this rule on every invocation.
%.pdf: %.tex
	$(LATEXMK) $(LATEXMKFLAGS) $<
	@touch $@

%.png: %.pdf
	$(MAGICK) -density $(DENSITY) $< $(MAGICKFLAGS) $@

clean:
	$(LATEXMK) -c

distclean:
	$(LATEXMK) -C
	$(RM) $(PNGS)
