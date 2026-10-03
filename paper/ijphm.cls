%% ijphm.cls -- LaTeX class for the International Journal of Prognostics and Health Management.
%% Adapted from the ERTS class (A. Albore, 2025), which is itself derived from the PHM Society LaTeX class
%% (J. Celaya, I. Roychoudhury, M. Daigle, 2012). Page layout, header/footer, licence note and citation
%% style follow the official IJPHM Word template (ijPHM-FullPaperTemplate.docx):
%%   US Letter; margins 0.75in left/right, 1.0in top/bottom; two columns, 0.25in apart;
%%   Times 10pt; unindented paragraphs, 6pt apart; APA citations; CC BY 3.0 US licence footnote.
%%
%% Options: year of publication, e.g. \documentclass[2026]{ijphm}.

\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{ijphm}[2026/10/03 IJPHM class adapted from the PHM Society class]

\def\publicationYear{20xx}
\DeclareOption{2026}{\def\publicationYear{2026}}
\DeclareOption{2027}{\def\publicationYear{2027}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[10pt,letterpaper]{article}

% ---------------------------------------------------------------------------------------------
% Packages
% ---------------------------------------------------------------------------------------------
\RequirePackage[T1]{fontenc}
\RequirePackage{times}
\RequirePackage{fancyhdr}
% flushend (column balancing on the last page) is omitted: it split the final reference across columns.
\RequirePackage[labelsep=period, format=plain, justification=centering]{caption}
\RequirePackage[hang,flushmargin]{footmisc}
\RequirePackage[natbibapa]{apacite}               % APA citations; \citep and \citet still work
\RequirePackage{setspace}
\RequirePackage{environ}
\RequirePackage[letterpaper, left=0.75in, right=0.75in, top=1in, bottom=1in, headsep=0.25in]{geometry}

% ---------------------------------------------------------------------------------------------
% Page layout
% ---------------------------------------------------------------------------------------------
\twocolumn
\setlength\columnsep{0.25in}
\flushbottom
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}

% ---------------------------------------------------------------------------------------------
% Header and footer: journal name on every page except the first; ISSN line on the first page
% ---------------------------------------------------------------------------------------------
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyhead[C]{\footnotesize\textsc{International Journal of Prognostics and Health Management}}
\fancyfoot[R]{\small\thepage}
\fancypagestyle{ijphmfirst}{%
  \fancyhf{}%
  \renewcommand{\headrulewidth}{0pt}%
  \fancyfoot[L]{\footnotesize International Journal of Prognostics and Health Management, ISSN 2153-2648, \publicationYear}%
  \fancyfoot[R]{\small\thepage}%
}

% ---------------------------------------------------------------------------------------------
% Title block: title 17pt bold, authors 11pt, affiliations and e-mails 9pt italic,
% followed by the one-column abstract
% ---------------------------------------------------------------------------------------------
\newcommand{\authorNumber}[1]{\textnormal{\textsuperscript{#1}}}
\newcommand{\affiliation}[3]{\authorNumber{#1}#2\\ #3}
\newcommand{\address}[1]{\def\@address{#1}}
\newcommand{\email}[1]{{\fontsize{9}{11}\selectfont\itshape #1}}
\def\@address{}

% The abstract is collected and typeset across both columns by \maketitle.
\def\@abstractbody{}
\RenewEnviron{abstract}{\global\let\@abstractbody\BODY}

\renewcommand{\maketitle}{%
  \twocolumn[{%
    \vspace*{0.25in}%
    \begin{center}
      {\fontsize{17}{20}\selectfont\bfseries\@title\par}
      \vskip 14pt
      {\fontsize{11}{13}\selectfont\@author\par}
      \vskip 11pt
      {\fontsize{9}{11}\selectfont\itshape
        \begin{tabular}[t]{c}\@address\end{tabular}\par}
    \end{center}
    \vskip 12pt
    \ifx\@abstractbody\@empty\else
      \centerline{\normalsize\bfseries\scshape Abstract}
      \vskip 6pt
      {\normalsize\noindent\@abstractbody\par}
    \fi
    \vskip 18pt
  }]%
  \thispagestyle{ijphmfirst}%
}

% ---------------------------------------------------------------------------------------------
% Licence footnote on the first page (CC BY 3.0 United States, as in the IJPHM template).
% Argument: first author's name, e.g. \licenseFootnote{Jane Doe et al.}
% ---------------------------------------------------------------------------------------------
\newcommand{\licenseFootnote}[1]{%
  \insert\footins{\noindent\footnotesize #1. This is an open-access article distributed under the terms of
    the Creative Commons Attribution 3.0 United States License, which permits unrestricted use,
    distribution, and reproduction in any medium, provided the original author and source are credited.}%
}

% ---------------------------------------------------------------------------------------------
% Sections: numbered "1." style; level 1 bold small capitals, levels 2-3 bold, all 10pt flush left
% ---------------------------------------------------------------------------------------------
\renewcommand\section{\@startsection{section}{1}{\z@}{12pt}{0.01pt}{\normalsize\bfseries\scshape}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}{6pt}{0.01pt}{\normalsize\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}{6pt}{0.01pt}{\normalsize\bfseries}}
\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}{6pt}{-0.5em}{\normalsize\bfseries}}
\renewcommand*{\@seccntformat}[1]{\csname the#1\endcsname.\quad}

% Unnumbered sections after the conclusion (acknowledgements, declarations, references).
\renewcommand\refname{References}

% Single-spaced bibliography with hanging indent.
\newcommand{\PHMbibliography}[1]{\begin{singlespace}\bibliography{#1}\end{singlespace}}

% ---------------------------------------------------------------------------------------------
% Floats, displays and lists (spacing from the PHM Society class)
% ---------------------------------------------------------------------------------------------
\footnotesep 6.65pt \skip\footins 9pt plus 4pt minus 2pt
\floatsep 12pt plus 2pt minus 2pt
\textfloatsep 16pt plus 2pt minus 4pt
\intextsep 12pt plus 2pt minus 2pt
\dblfloatsep 12pt plus 2pt minus 2pt
\dbltextfloatsep 18pt plus 2pt minus 4pt
\abovedisplayskip 7pt plus2pt minus5pt%
\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip 0pt plus3pt%
\belowdisplayshortskip 4pt plus3pt minus3pt%
\leftmargini 0.25in
\leftmarginii 2em
