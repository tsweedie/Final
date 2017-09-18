%% Abdul Rahman Mojadadi
%% This is file ``uwcthesis.cls'', for use at The University of the Western Cape
%% It is based on the standard LaTeX report class.
%% Originally written by Harriet Borton (ITS), April 1996.
%% Modified April 11, 1996; minor revisions February, 2001.
%%          2016-02-25      chapter left justified
%%                          Adapted for Science Faculty UWC
%% 
%% The SunLab machines have template files for preparing a thesis 
%% in the directory:
%%
%% /export/home/notes/latex/uwcthesis/
%% 
%% This file is part of the LaTeX2e system. 
%% ---------------------------------------- 
%% 
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{uwchonoursthesis}[2016/02/25 University of the Western Cape]
%  Note that the setspace package is built in (code included near the end of 
%  this file) to provide "line-and-a half spacing" (1.4 by default) and also
%  the singlespace environment. 

% UWC option chapter
\newif\ifchap  % true for chapter option
  \chapfalse   % false by default
\DeclareOption{chapter}{\chaptrue} % option to print "Chapter" 
%                                  % left-justified at each chapter
\newcommand\docsize{}  % to allow 10pt or 11pt to be specified as option
\DeclareOption{10pt}{\renewcommand\docsize{10pt}}
\DeclareOption{11pt}{\renewcommand\docsize{11pt}}
\DeclareOption{12pt}{\renewcommand\docsize{12pt}}
%  Prepare to load the standard report class (12pt):
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ExecuteOptions{12pt}         % define 12pt as the default doc size
\ProcessOptions
\LoadClass[\docsize]{report}  % load report.cls
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%  The following sections are revisions or additions to report.cls
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Footnotes:   make them continuously numbered throughout document
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define command that can undo having footnotes reset with each chapter:
% (taken from removefr.sty by Donald Arseneau) 
\def\@removefromreset#1#2{\let\@tempb\@elt
   \expandafter\let\expandafter\@tempa\csname c@#1\endcsname
   \def\@elt##1{\expandafter\ifx\csname c@##1\endcsname\@tempa\else
         \noexpand\@elt{##1}\fi}%
   \expandafter\edef\csname cl@#2\endcsname{\csname cl@#2\endcsname}%
   \let\@elt\@tempb}
% use the command \@removefromreset to undo the \@addtoreset in report.cls:
\@removefromreset{footnote}{chapter}
%
% define command to allow people to reset footnote counter at will: 
\def\resetfootnote{\setcounter{footnote}{0}}  % definition to reset footnote


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Page Layout
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Side Margins:
\if@twoside                 % Values for two-sided printing:
   \oddsidemargin .55in     %   Left margin on odd-numbered pages.
   \evensidemargin .05in    %   Left margin on even-numbered pages.
   \marginparwidth 40pt     %   Width of marginal notes.
\else                       % Values for one-sided printing:
   \oddsidemargin 0.55in    %   Note that \oddsidemargin = \evensidemargin
   \evensidemargin 0.55in
   \marginparwidth 00pt
\fi
\hoffset -20pt
\marginparsep 10pt          % Horizontal space between outer margin and
                            % marginal note                 A4 8.29 - 8.50
\textwidth 5.48in           % width of letter text 5.9in    A4 text 5.48in
%\textwidth 420pt            % width of letter text 5.9in    A4 text 5.48in
                           
% VERTICAL SPACING:        
                            % Top of page:
\topmargin -27pt            %   distance from top of page to running head
%\topmargin -.5in           %   distance from top of page to running head
\headheight 14pt            %   Height of box containing running head.
\headsep .4in               %   Space between running head and text.
%\textheight 684pt           %   space for letter text 8.8in and A4 text 9.49in 
\textheight 9.49in          %   space for letter text 8.8in and A4 text 9.49in 
\footskip 40pt              %   Distance from baseline of box containing foot
                            %   to baseline of last line of text.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Section Headings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
\newcommand\chaptersize{\LARGE}
\newcommand\sectionsize{\large}
\newcommand\subsectionsize{\normalsize}
\newcommand\subsubsectionsize{\normalsize}
\newcommand\subsubsubsectionsize{\normalsize}
\newcounter{firstchapter}
\setcounter{firstchapter}{0}

\setcounter{secnumdepth}{3}    % Number subsubsections in the chapters
\setcounter{tocdepth}{2}       % Put subsubsections in the table of contents

% Print "Chapter" if chap option is specified:
\ifchap
  \renewcommand\@chapapp{\chaptername}
\else
  \renewcommand\@chapapp{}
\fi

\def\specialhead#1{%   For headings like Abstract, Key Words, etc.
     \newpage
     \chapter*{%\centering 
               \leftline{#1 \hfil} \@mkboth{#1}{#1}}
     \addcontentsline{toc}{chapter}{#1}}

\def\specialnohead#1{%   For headings like Abstract, Key Words, etc.
%     \newpage
%     \chapter*{%\centering 
%               \leftline{#1 \hfil} \@mkboth{#1}{#1}}
     \addcontentsline{toc}{chapter}{#1}}

\def\uwcbibliography#1{%   For headings like Abstract, Key Words, etc.
     %\newpage
     %\chapter*{%\centering 
     %          \leftline{#1 \hfil} \@mkboth{#1}{#1}}
     \addcontentsline{toc}{chapter}{Bibliography}
     \bibliography{#1}}

\def\@chapter[#1]#2{\ifnum\c@firstchapter=0    % ex-RPI: start of rpi added stuff
                      \if@twoside\cleardoublepage\suppressfloats[t]\fi
                      \pagenumbering{arabic} 
                      \setcounter{firstchapter}{1}
                    \fi
                    \renewcommand{\thepage}{\arabic{page}}
                    \thispagestyle{plain}
                    \pagestyle{myheadings}      % ex-RPI end of rpi added stuff
                    \ifnum \c@secnumdepth >\m@ne
                         \refstepcounter{chapter}%
                         \typeout{\@chapapp\space\thechapter.}%
                         \addcontentsline{toc}{chapter}%
                                   {\protect\numberline{\thechapter.}#1}%
                    \else
                      \addcontentsline{toc}{chapter}{#1}%
                    \fi
                    \chaptermark{#1}% 
%                    \addtocontents{lof}{\protect\addvspace{10\p@}}%
%                    \addtocontents{lot}{\protect\addvspace{10\p@}}%
                    \if@twocolumn
                      \@topnewpage[\@makechapterhead{#2}]%
                    \else
                      \@makechapterhead{#2}%
                      \@afterheading
                    \fi}
\def\@makechapterhead#1{%
  \vspace*{0\p@}%
  {\parindent \z@ \raggedright %\centering 
                                           \normalfont \chaptersize
    \ifnum \c@secnumdepth >\m@ne
      \ifchap
         \bfseries \@chapapp{} \thechapter    % print "Chapter" and number
         \vskip +24pt           %\par\nobreak (original)
       \else
         \bfseries \thechapter. 
       \fi
    \fi
    \interlinepenalty\@M
    \bfseries #1\par\nobreak
    \vskip 15\p@
  }
    \noindent}

\def\@makeschapterhead#1{%    heading for chapter* command (no numbering)
  \vspace*{0\p@}%
  {\parindent \z@ \raggedright %\centering
    \normalfont  \chaptersize 
    \interlinepenalty\@M
    \bfseries  #1\par\nobreak
    \vskip 15\p@
  }}

\renewcommand\section{\@startsection {section}{1}{\z@}%
                                   {3.5ex \@plus 1ex \@minus .2ex}%
                                   {.5ex \@plus .3ex}%{1.4ex \@plus.2ex}%
                                   {\normalfont\sectionsize\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {3.25ex\@plus 1ex \@minus .2ex}%
                                     {.3ex \@plus .2ex}%{1.2ex \@plus .2ex}%
                                     {\normalfont\subsectionsize\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {3.25ex\@plus 1ex \@minus .2ex}%
                                     {.2ex \@plus .1ex}%{1ex \@plus .2ex}%
                                     {\normalfont\subsubsectionsize\bfseries}}
%\renewcommand\subsubsubsection{\@startsection{subsubsubsection}{3}{\z@}%
%                                     {3.25ex\@plus 1ex \@minus .2ex}%
%                                     {.2ex \@plus .1ex}%{1ex \@plus .2ex}%
%                                     {\normalfont\subsubsubsectionsize\bfseries}}
% \paragraph and \subparagraph headings unchanged from report.cls.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Appendix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\renewcommand\appendix{\par
  \setcounter{chapter}{0}%
  \setcounter{section}{0}%
  \chaptrue
  \renewcommand\@chapapp{\appendixname}%
  \renewcommand\thechapter{\@Alph\c@chapter}}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Figures and Tables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%control float placement:
%
\setcounter{topnumber}{2}
\renewcommand\topfraction{.8}
\setcounter{bottomnumber}{2}
\renewcommand\bottomfraction{.8}
\setcounter{totalnumber}{4}
\renewcommand\textfraction{.2}
\renewcommand\floatpagefraction{.8}
\setcounter{dbltopnumber}{2}
\renewcommand\dbltopfraction{.8}
\renewcommand\dblfloatpagefraction{.8}

\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{{\bfseries#1}:
                  ~~#2}%       % Make caption normal
  \ifdim \wd\@tempboxa >\hsize
   {{\bfseries#1}:                % Make caption bold 
     ~~#2}\par%    
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Table of Contents,  List of Tables,  List of Figures 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\renewcommand\tableofcontents{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\leftline{\contentsname \hfil}
        \@mkboth{%
           \MakeUppercase\contentsname}{\MakeUppercase\contentsname}}%
    \vskip -1em \begin{singlespace}      % singlespacing
    \@starttoc{toc}%
    \if@restonecol\twocolumn\fi
    \end{singlespace}
    }
 
\renewcommand*\l@chapter{\pagebreak[3]\vskip 10pt plus 1pt minus 1pt
%                              \@dottedtocline{3}{7.0em}{4.1em}}
                           \@dottedtocline{0}{0em}{3.2em}}
%                         \@dottedtocline{0}{0em}{1.4em}}
\renewcommand*\l@section{\vskip 6pt plus 1pt minus 1pt
%                              \@dottedtocline{3}{7.0em}{4.1em}}
                           \@dottedtocline{2}{0em}{3.2em}}
%                        \@dottedtocline{1}{1.5em}{2.3em}}
\renewcommand*\l@subsection{\ifnum\c@tocdepth>1\vskip 4pt minus 1pt \fi
%                              \@dottedtocline{3}{7.0em}{4.1em}}
                           \@dottedtocline{0}{0em}{3.2em}}
%                           \@dottedtocline{2}{3.8em}{3.2em}}
\renewcommand*\l@subsubsection{\ifnum\c@tocdepth>2 \vskip 3pt minus 1pt \fi
%                               \@dottedtocline{3}{7.0em}{4.1em}}
                           \@dottedtocline{0}{0em}{3.2em}}
%                              \@dottedtocline{3}{7.0em}{4.1em}}

% The following removed because it's not consistent with entries from longtable
%% modify the definition below (taken from latex.ltx) to include 
%% "Table" and "Figure" in entries for lot and lof: 
%\long\def\@caption#1[#2]#3{%\baselineskip 14.5 pt
%  \addcontentsline{\csname ext@#1\endcsname}{#1}%
%  {\protect\numberline{\csname fnum@#1\endcsname}{\ignorespaces #2}}%
%  \begingroup
%    \@parboxrestore
%    \normalsize
%    \@makecaption{\csname fnum@#1\endcsname}{\ignorespaces #3}\par
%  \endgroup}
 
\renewcommand\listoffigures{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{%\centering\listfigurename  % center it
               \leftline{\listfigurename \hfil} % left justify it
      \@mkboth{\MakeUppercase\listfigurename}%
              {\MakeUppercase\listfigurename}}%
    \addcontentsline{toc}{chapter}{\listfigurename}  % add lof to toc
    \vskip -1em \begin{singlespace}  % singlespacing
    \@starttoc{lof}%
    \if@restonecol\twocolumn\fi
    \end{singlespace}
    }

\renewcommand\listoftables{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{%\centering \listtablename  % center it
              \leftline{\listtablename \hfil} % left justify it
      \@mkboth{\MakeUppercase\listtablename}%
              {\MakeUppercase\listtablename}}%
    \addcontentsline{toc}{chapter}{\listtablename}  % add lot to toc
    \vskip -1em \begin{singlespace}  % singlespacing
    \@starttoc{lot}%
    \if@restonecol\twocolumn\fi
    \end{singlespace}
    }

%\renewcommand\bibliography[1]{%
%    \if@twocolumn
%      \@restonecoltrue\onecolumn
%    \else
%      \@restonecolfalse
%    \fi
%    \chapter*{%\centering \listtablename  % center it
%              \leftline{\bibname \hfil} % left justify it
%      \@mkboth{\MakeUppercase\bibname}%
%              {\MakeUppercase\bibname}}%
%    \addcontentsline{toc}{chapter}{\bibname}  % add lot to toc
%    \vskip -1em \begin{singlespace}  % singlespacing
%    \@starttoc{lot}%
%    \if@restonecol\twocolumn\fi
%    \end{singlespace}
%    #1
%    }
%
%\renewcommand{\thebibliography}[1]{%
%   \clearpage   % necessary to get correct page number in toc
%   \ifdal@print@bib
%      \addcontentsline{toc}{chapter}{\bibname}
%      \linespread{1}\normalsize
%      \orig@thebibliography{#1}
%   \else
%      \newbox\dal@bib@box
%      \newbox\dal@bibitem@box
%      \let\b@group={
%      \let\e@group=}
%      \def\endthebibliography{\egroup\e@group}
%      \let\orig@bibitem\bibitem
%      \def\bibitem{%
%         \ifdal@first@bibitem
%            \dal@first@bibitemfalse
%         \else
%            \egroup
%         \fi
%         \setbox\dal@bibitem@box=\vbox\bgroup
%         \orig@bibitem}
%      \setbox\dal@bib@box=\vbox\b@group
%      \if@filesw
%         \immediate\write\@auxout{\string\@writefile{toc}%
%            {\string\contentsline \space{chapter}{\bibname}{\thepage}}}
%      \fi
%      \orig@thebibliography{#1}
%   \fi
%   }
%

%remove following at same time as remove code to put "Figure" in LOF
%\renewcommand*\l@figure{\vskip 10pt plus 1pt minus 1pt
%                       \@dottedtocline{1}{0em}{5.8em}}
 
\renewcommand*\l@figure{\vskip 10pt plus 1pt minus 1pt
                       \@dottedtocline{1}{0em}{2.8em}}
\let\l@table\l@figure

\let\rpicaption\caption
\let\lrpicaption\caption


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Bibliography
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RPI def: for use in making an unnumbered bibliography with hanging indents
\def\bibentry{\vskip10pt\par\noindent\hangindent=40pt\frenchspacing}

% report definition modified for no automatic heading and use ragged right
\newcommand{\bibalign}{\raggedright}

\renewenvironment{thebibliography}[1]
     {\bibalign\frenchspacing
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy\clubpenalty4000\widowpenalty4000%
      \sfcode`\.=\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  FOR UWC titlepage, declaration, abstract, keywords
%  UWC does not allow a copyright page
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 \def\thesistitle#1{\gdef\@thesistitle{#1}}
 \def\author#1{\gdef\@author{#1}}
 \def\degree#1{\gdef\@degree{#1}}
 \def\department#1{\gdef\@department{#1}}
 \newcount\numcomm \numcomm=1
 \def\signaturelines#1{\numcomm=#1}
 \def\supervisor#1{\gdef\@supervisor{Supervisor: #1}} 
 \def\cosupervisor#1{\gdef\@cosupervisor{Co-supervisor: #1}} 
 \def\thadvisor#1{\gdef\@thadvisor{#1}} 
 \def\projadvisor#1{\gdef\@projadvisor{#1}} \projadvisor{*}
 \def\cothadvisor#1{\gdef\@cothadvisor{#1}} \cothadvisor{*} 
 \def\coprojadvisor#1{\gdef\@coprojadvisor{#1}} \coprojadvisor{*}
 \def\doctype{\if \@projadvisor *Thesis \else Project \fi}
 \def\advisor{\if \@projadvisor *\@thadvisor \else \@projadvisor\fi}
 \def\coadvisor{\if \@coprojadvisor *\@cothadvisor\else \@coprojadvisor\fi}
 \def\memberone#1{\gdef\@memberone{#1}}
 \def\membertwo#1{\gdef\@membertwo{#1}}
 \def\memberthree#1{\gdef\@memberthree{#1}}
 \def\memberfour#1{\gdef\@memberfour{#1}}
 \def\memberfive#1{\gdef\@memberfive{#1}}
 \def\membersix#1{\gdef\@membersix{#1}}
 \def\copyrightyear#1{\gdef\@copyrightyear{#1}} \copyrightyear{\the\year}
 \def\submitdate#1{\gdef\@submitdate{#1}}

\newcount\numcount
\def\supervisorpromotor{
     \vskip .8in plus 24pt minus 24pt
       \@supervisor 
      \vskip 1pt
       \@cosupervisor}

\def\sigblock{
   \ifnum\numcomm < 3  % master's thesis or project
     \vskip .8in plus 24pt minus 24pt
     \if \coadvisor *  % no coadvisor
       \leftline{Approved: \hfil}\vskip 48pt
       \hrule width2.8in\hfil \vskip -3pt
       \leftline{\advisor \hfil}
       \leftline{\doctype Advisor \hfil}
       \vskip 6pt 
     \else
       \leftline{Approved: \hfil}\vskip 48pt
       \hrule width2.8in\hfil \vskip -3pt
       \leftline{\advisor, \doctype Advisor\hfil}
%      \leftline{\advisor, Co-\doctype Advisor\hfil}   % this puts in "Co"
       \vskip 36pt  \hrule width 2.8in\hfil \vskip -3pt
       \leftline{\coadvisor, \doctype Advisor\hfil}
%      \leftline{\coadvisor, Co-\doctype Advisor\hfil} % this puts in "Co"
       \vskip 6pt       
     \fi
   \else  % PhD thesis
      \numcount=\numcomm
      \leftline{Approved by the\hfil}
      \leftline{Examining Committee:\hfil}
      \vskip 28pt \vfil 
      \hrule width 2.8in \hfil \vskip -3pt
      \leftline{\advisor, \doctype Advisor\hfil} 
%     \leftline{\advisor, Co-\doctype Advisor\hfil} % put "Co-" in the line
      \advance \numcount by -1
      \if \coadvisor * \else % there's a co-advisor
        \vskip 19pt\vfil\hrule width 2.8in\hfil\vskip -3pt
        \leftline{\coadvisor, \doctype Advisor\hfil}
%       \leftline{\coadvisor, Co-\doctype Advisor\hfil} % put "Co-" in the line
        \advance \numcount by -1
      \fi
      \vskip 19pt\vfil\hrule width 2.8in\hfil\vskip -3pt
      \leftline{\@memberone, Member\hfil}
      \advance \numcount by -1
      \vskip 19pt\vfil\hrule width 2.8in\hfil\vskip -3pt
      \leftline{\@membertwo, Member\hfil}
      \advance \numcount by -1 \ifnum\numcount > 0
      \vskip 19pt\vfil\hrule width 2.8in\hfil\vskip -3pt
      \leftline{\@memberthree, Member\hfil}  \fi
      \advance \numcount by -1 \ifnum\numcount > 0
      \vskip 19pt\vfil\hrule width 2.8in\hfil\vskip -3pt
      \leftline{\@memberfour, Member\hfil}   \fi
      \advance \numcount by -1 \ifnum\numcount > 0
      \vskip 19pt\vfil\hrule width 2.8in\hfil\vskip -3pt
      \leftline{\@memberfive, Member\hfil}  \fi
      \advance \numcount by -1 \ifnum\numcount > 0
      \vskip 19pt\vfil\hrule width 2.8in\hfil\vskip -3pt
      \leftline{\@membersix, Member\hfil} \fi
      \fi} 

 \def\sigblockabs{%
    \numcount=\numcomm
    \leftline{Examining Committee: \hfil}
    \begin{quote}   
      \advisor, \doctype Advisor\\[4pt]  \advance \numcount by -1
%     \advisor, Co-\doctype Advisor\hfil\\[4pt]  \advance \numcount by -1
      \if \coadvisor * \else 
        \coadvisor, \doctype Advisor\\[4pt]    \advance\numcount by -1
%       \coadvisor, Co-\doctype Advisor\\[4pt] \advance\numcount by -1
      \fi
      \@memberone, Member\\[4pt] \advance \numcount by -1
      \ifnum\numcount > 0
      \@membertwo, Member\\[4pt] \advance \numcount by -1 \fi
      \ifnum\numcount > 0
      \@memberthree, Member\\[4pt] \advance \numcount by -1 \fi
      \ifnum\numcount > 0
      \@memberfour, Member\\[4pt] \advance \numcount by -1  \fi
      \ifnum\numcount > 0
      \@memberfive, Member\\[4pt]\advance \numcount by -1  \fi
      \ifnum\numcount > 0
      \@membersix, Member\\[4pt] \advance \numcount by -1\fi
    \end{quote}
 }

 \def\titlepage{%
     \thispagestyle{empty}
     \ifnum\numcomm<3 \vglue .5in\vfil \fi  % masters
     \ifnum\numcomm>5 \hbox{ } \vspace{-24pt}\fi % make more space on page
     \begin{singlespace}
     \begin{center}
         \newlength\oldbaseline
         \parindent=0pt
         \setlength\oldbaseline{\baselineskip}%
         \setlength\baselineskip{24pt}%
         {\LARGE % \Huge % \uppercase%
                \expandafter{\bf \@thesistitle}}\\ [96pt]
         \setlength\baselineskip{\oldbaseline}%
         \@author\\ [192pt]
         \doctype presented in fulfilment \\ %[8pt]
         of the requirements for the degree of \\ %[8pt]%\uppercase%
         \expandafter{\@degree}\\ %[8pt]
         at the University of the Western Cape \\ [36pt]
         %\ifnum \numcomm > 2 Major Subject:~~\@department\\  \fi
         %\ifnum\numcomm < 7 \vskip 32pt \else \vskip 24pt \fi
         %\sigblock 
         %\ifnum \numcomm < 7 \vskip 32pt \else \vskip 24pt \fi
         %University of the Western Cape\\
         %Bellville, Cape Town\\ [72pt]
         
         \ifnum\numcomm>6 \vskip -8pt \fi
         \supervisorpromotor
         \vskip +12pt
         \@submitdate
     \end{center}
     \end{singlespace}
     \pagestyle{plain}
     \pagenumbering{roman}
     \setcounter{page}{1}
     \openpage
     \newpage
     \declaration
     \openpage
     }
 

\def\openpage{%
	\vspace{5cm}
   \mbox{ }
   \pagebreak
   }


\def\abstract{%
	\specialhead{Abstract}
	\vspace{5cm}
   }


\def\keywords{%
	\specialhead{Key words}
	\vspace{5cm}
   }

\def\acknowledgements{%
	\specialhead{Acknowledgements}
	\vspace{5cm}
   }


\def\declaration{%
	\specialhead{Declaration}
	\vspace{5cm}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                 %
%                         Declaration                             %
%                                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\vfill\vfill
I, \textsc{\@author}, declare that this thesis ``\textit{\@thesistitle}'' is  my 
own work, that it has not been submitted before for any degree or 
assessment at any other university, and that all the sources I have 
used or quoted have been indicated and acknowledged by means of 
complete references.
\\*[+24pt]
	%\end{minipage}
	%\end{center}
	\begin{center}
	  Signature:\ \ \ldots\ldots\ldots\ldots\ldots\ldots\ldots\ldots
	  \hfill
	  Date:\ \ \ldots\ldots\ldots\ldots\ldots\ldots\ldots\ldots\\
     \textsc{\@author}.\hfill\hfill\hfill\hfill

	\end{center}
\pagebreak}


\def\preface{%
	\specialhead{Preface}
	\vspace{5cm}
   }


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The CODE FROM SETSPACE.STY:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\setstretchbl}[1]{%
  \def\baselinestretch{#1}%
  \@currsize
}
%
%  Has been removed
%
%%%%%%%%%%%%%%%%%%%%%%  End of setspace.sty %%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Some Initializations:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% make the following names uppercase:
\renewcommand\contentsname{Contents}
\renewcommand\listfigurename{List of Figures}
\renewcommand\listtablename{List of Tables}
\renewcommand\bibname{Bibliography}
\renewcommand\indexname{Index}
\renewcommand\partname{Part}
\renewcommand\chaptername{Chapter}
\renewcommand\appendixname{Appendix}
\renewcommand\abstractname{Abstract}

\renewcommand\chaptersize{\LARGE}
\renewcommand\sectionsize{\large}
\renewcommand\subsectionsize{\normalsize}
\renewcommand\subsubsectionsize{\normalsize}
\renewcommand\subsubsubsectionsize{\normalsize}

\raggedbottom

\setstretchbl{1.4}        % at 11 or 12pt this is line-and-a-half spacing
\parindent .4in         % Width of paragraph indentation

\markboth{}{}           % Do not include chapter titles in headers;
\pagestyle{myheadings}  % include just page numbers in upper right

\renewcommand{\bibalign}{\raggedright}  %  bibliography is ragged right
% for fully justified bibliography: \renewcommand{\bibalign}{}

%% Force \noindent after \section, \subsection, etc.
%%       The default of report.cls that does this automatically has 
%%       been clobbered and I did not bother to try find the smash.
\let\@afterindenttrue\@afterindentfalse
\@afterindentfalse

\endinput
%% 
%% End of file `uwcthesis.cls'.
