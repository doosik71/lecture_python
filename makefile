SHELL = /bin/bash

home_path := $(abspath .)
notebook_files := $(wildcard */*.ipynb)
pdf_files := $(patsubst %.ipynb, pdf/%.pdf, $(notebook_files))
slide_files := $(patsubst %.ipynb, slides/%.slides.html, $(notebook_files))
pdf_path := $(abspath pdf)
png_images := $(patsubst %, slides/%, $(wildcard ??_*/*.png))
gif_images := $(patsubst %, slides/%, $(wildcard ??_*/*.gif))
jpg_images := $(patsubst %, slides/%, $(wildcard ??_*/*.jpg))

all: ${pdf_files} ${slide_files} ${png_images} ${gif_images} ${jpg_images}

# PDFs
pdf/%.pdf: %.ipynb
	cd "$(dir $*.ipynb)" ; pandoc --pdf-engine=xelatex --variable mainfont="Noto Serif CJK SC" "$(notdir $*.ipynb)" -o "${pdf_path}/$*.pdf"

# SLIDEs
slides/%.slides.html: %.ipynb
	jupyter nbconvert --to=slides --output-dir="slides/$(dir $*.slides.html)" $*.ipynb

# IMAGEs
slides/%.png: %.png
	cp $*.png $(dir ./slides/$*.png)

slides/%.gif: %.gif
	cp $*.gif $(dir slides/$*.gif)

slides/%.jpg: %.jpg
	cp $*.jpg $(dir slides/$*.jpg)
