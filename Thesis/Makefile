#
# Makefile - makes latex-slides.ps, creates a 4up PostScript copy and 
#            corrects the access codes, copies the files to the
#            web pages in ~/www/latex/
#
#Put your surnameFirstName here, e.g. doddsReg 

D=regdodds
D=tshidilehata
Dps=$(D).ps
Dtex=$(D).tex
WWW=../../www/latex/

all:	$(Dps)

$(Dps):	$(D).dvi 
#	dvips -o $(D).ps $(D).dvi   #This creates a ps version of thesis from dvi file
#  pstops creates the 4up-ps 
#	pstops '4:0@.55(0,14cm)+1@.55(9.5cm,14cm)+2@.55(0,0)+3@.55(9.5cm,0)' $(Dps) 4up-$(Dps)
#  creates pdf from dvi 
#	dvipdf $(D).dvi             #This creates a pdf version of thesis from dvi file
#	chmod 644 $(Dps)
#	chmod 644 4up-$(Dps)
#	chmod 644 $(Dtex)
#  Uncommented this moves ps and 4up-ps pages to web location at $(WWW)
#	-cp 4up-$(Dps) $(WWW)
#	-cp $(Dps) $(WWW)
#	-cp $(Dtex) $(WWW)
#	gv $(Dps)
#	make clean
#	latex $(D)
#	dvips $(D).dvi
#	dvipdf $(D)

$(D).dvi: $(D).tex *chapter.dvi bibliography.dvi appendix*.dvi
	make clean
	echo "$D.tex $(D).dvi $(D).tex"
	-rm -f *.aux
	pdflatex -shell-escape $(D)
	bibtex $(D)
	makeindex $(D)
	pdflatex -shell-escape $(D)
	pdflatex -shell-escape $(D)
#	latex $(D)
#	latex $(D)
#	latex $(D)

*chapter.dvi: *chapter.tex 
	echo "$D.tex $(D).dvi $(D).tex"
	echo "dummy operation on $(@)"

bibliography.dvi: acknowledgements.tex
	echo "$D.tex $(D).dvi $(D).tex"
	echo "dummy operation on $(@)"

bibliography.dvi: bibliography.tex
	echo "$D.tex $(D).dvi $(D).tex"
	echo "dummy operation on $(@)"

appendix*.dvi: appendix*.tex
	echo "$D.tex $(D).dvi $(D).tex"
	echo "dummy operation on $(@)"

clean:
	-rm -f $(D).aux
	-rm -f *chapter.aux
	-rm -f *.aux
	-rm -f $(D).bbl
	-rm -f $(D).blg
	-rm -f $(D).cre
	-rm -f $(D).crf
	-rm -f $(D).dvi
	-rm -f $(D).idx
	-rm -f $(D).ilg
	-rm -f $(D).ind
	-rm -f $(D).lof
	-rm -f $(D).log
	-rm -f $(D).lot
#	-rm -f $(D).log
	-rm -f $(D).not
	-rm -f $(D).cb
	-rm -f $(D).ptc
	-rm -f 4up-$(D).ps
	-rm -f 4up-$(D).pdf
	-rm -f $(D).pdf
	-rm -f $(D).toc 
	-rm les*.aux
	-rm *~
	-rm *.bak
	-rm figures/*.bak
	-mkdir ../backup
	-rm -r backup
	-cp -R * ../backup/.
	-mv ../backup .

cleanup:
	-rm -f $(D).pdf
	-rm -f $(D).ps
	-make clean
	-rm -rf backup

cleanall:
	-make clean
	-rm -rf backup

