%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Engineer & Master Thesis LaTeX Template Class     %%
%% Copyleft by Piotr Wozniak & Artur M. Brodzki      %%
%% Faculty of Electronics and Information Technology %%
%% Warsaw University of Technology, Warsaw, 2019     %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\ProvidesClass{eiti/eiti-thesis}

\LoadClass[
    12pt, 
    oneside
]{article}

%\DeclareMathSizes{12}{14}{11}{10}     % For size 12 text
%--------------------------------------
% Define class parameters
%--------------------------------------
\RequirePackage{kvoptions}
\SetupKeyvalOptions{
    family=eiti/eiti-thesis.cls,
    prefix=eiti
}

\DeclareStringOption[2.5cm]{left}
\DeclareStringOption[2.5cm]{right}
\DeclareStringOption[2.5cm]{top}
\DeclareStringOption[2.5cm]{bottom}
\DeclareStringOption[6mm]{bindingoffset}
\DeclareBoolOption[false]{nohyphenation}
\ProcessKeyvalOptions*

%--------------------------------
% Common packages
%--------------------------------
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{amsthm}
\RequirePackage{array} 
\RequirePackage[
    labelfont=bf,
    labelsep=period
]{caption}
\RequirePackage{enumitem}
\RequirePackage{float}

\RequirePackage{mathpazo}
\usepackage{mathpple}
%\RequirePackage{cmbright}
%\RequirePackage{euler}

%\RequirePackage{fancyhdr}
%\usepackage{millennial}
%\usepackage{fouriernc}
%\RequirePackage{fourier}
%\usepackage{fix-cm, fourier}
%\def\fourierScale{s*[1.2]}   


\RequirePackage{graphicx}
%\RequirePackage{ifluatex}
\RequirePackage{ifpdf}
%\RequirePackage{ifxetex}
%\RequirePackage{kantlipsum}
%\RequirePackage{lipsum}
\RequirePackage{listings}
\RequirePackage{longtable}
\RequirePackage{multirow}
\RequirePackage{setspace}
%\RequirePackage{tgadventor}
\RequirePackage{url}
\RequirePackage{xkeyval}
\RequirePackage{xspace}

%--------------------------------------
% PdfTeX specific configuration
%--------------------------------------
\ifpdf
    \RequirePackage[T1]{fontenc}
    \RequirePackage[utf8]{inputenc}
    \RequirePackage[
        protrusion=true,
        expansion=true
    ]{microtype}
\fi

%--------------------------------
% Define hyperref colors
%--------------------------------
\RequirePackage{hyperref}
\hypersetup{
    colorlinks,
    citecolor=black,
    filecolor=black,
    linkcolor=black,
    urlcolor=black
}

%--------------------------------------
% Default margin is 2.5 cm
%--------------------------------------
\RequirePackage[
    a4paper, 
    left=\eitileft,
    right=\eitiright,
    top=\eititop,
    bottom=\eitibottom,
    bindingoffset=\eitibindingoffset
]{geometry}

%--------------------------------
% Turn off hyphenation
%--------------------------------
\ifeitinohyphenation
    \tolerance=1
    \emergencystretch=\maxdimen
    \hyphenpenalty=10000
    \hbadness=10000
\fi

%--------------------------------
% Initial setup
%--------------------------------
\setstretch{1.2}
%\SetSectionFormatting{section}{0.5cm}{\FormatHangHeading{\Large}}{0.5cm}
\let\oldsection\section
\renewcommand{\section}{
    \thispagestyle{plain}
    \oldsection
}
%\fancypagestyle{plain}{
    %\fancyhf{}
    %\renewcommand{\headrulewidth}{0pt}
    %\fancyfoot[LE,RO]{\thepage}
%}

%--------------------------------
% Header & footer
%--------------------------------
%\pagestyle{fancy}
%\renewcommand{\sectionmark}[1]{
    %\markboth{#1}{#1}
%}
%
%\fancyhead{}
%\fancyheadoffset{0cm} 
%\fancyhead[RO]{\nouppercase{\thesection.\hspace{8pt}\leftmark}}
%\fancyhead[LE]{\nouppercase{\thesection.\hspace{8pt}\leftmark}}
%\fancyfoot{}
%\fancyfoot[LE,RO]{\thepage}

%--------------------------------
% Set default list style
%--------------------------------
\setlist[itemize,1]{topsep=2pt,label=\large$\bullet$, leftmargin=28pt}
\setlist[itemize,2]{topsep=2pt,leftmargin=18pt}
\setlist[itemize,3]{topsep=2pt,leftmargin=18pt}

\setlist[enumerate,1]{topsep=2pt,leftmargin=28pt}
\setlist[enumerate,2]{topsep=2pt,leftmargin=16pt}
\setlist[enumerate,3]{topsep=2pt,leftmargin=16pt}

%--------------------------------
% Streszczenie po polsku
%--------------------------------
\newcommand{\resumen}{
    \thispagestyle{empty}
    \begin{center}\textbf{\large\@title}\end{center}
    \textbf{\\ Resumen.\xspace}
}
\newcommand{\palabrasclave}{\vspace{0.5cm}\par\noindent \textbf{Palabras clave: \xspace}}

%--------------------------------
% Streszczenie po angielsku
%--------------------------------
\renewcommand{\abstract}{
    \thispagestyle{empty}
    \begin{center}\textbf{\large\@engtitle}\end{center}
    \textbf{\\ Abstract.\xspace}
}
\newcommand{\keywords}{\vspace{0.5cm}\par\noindent \textbf{Keywords: \xspace}}

%--------------------------------
% Oświadczenie o autorstwie
%--------------------------------
\newcommand{\makeauthorship}{
    \thispagestyle{empty}
    \begin{figure}[H]
        \vspace{-55pt}
        \noindent\makebox[\textwidth]{
            \includegraphics[width=1.3\textwidth]{eiti/oswiadczenie_autora_pracy.pdf}
        }
    \end{figure}
}

%--------------------------------
% Strona tytułowa - makra
%--------------------------------
\newcommand{\EngineerThesis}{
    \newcommand{\ThesisType}{Engineer}
}
\newcommand{\MasterThesis}{
    \newcommand{\ThesisType}{Master}
}
\newcommand{\instytut}[1]{
    \newcommand{\@instytut}{#1}
}
\newcommand{\kierunek}[1]{
    \newcommand{\@kierunek}{#1}
}
\newcommand{\specjalnosc}[1]{
    \newcommand{\@specjalnosc}{#1}
}
\newcommand{\album}[1]{
    \newcommand{\@album}{#1}
}
\newcommand{\promotor}[1]{
    \newcommand{\@promotor}{#1}
}
\newcommand{\engtitle}[1]{
    \newcommand{\@engtitle}{#1}
}

%--------------------------------
% Strona tytułowa
%--------------------------------
\let\oldmaketitle\maketitle
\renewcommand{\maketitle}{
    \linespread{1.15}
    
    \thispagestyle{empty}
    \pagenumbering{gobble}
    
    \begin{center}
        %\sffamily
        \includegraphics[width=0.8\textwidth]{eiti/fiuba_header.jpg} \\
        %\hfill \break
        %\hfill \break
        %Instytut \@instytut \\
        %\hfill \break
        %\hfill \break
        \hfill \break
        \LARGE
            Propuesta de tesis de grado de 
            Ingeniería en Informática%\includegraphics[width=\textwidth]{eiti/title-inz.png} \\
    
        
        \hfill \break
        \normalsize
        Director: Dr. Leandro Lombardi  \\
        Co-Director: Lic. Luis Argerich \\
        Alumno: Joaquín Torré Zaffaroni \\
        \hfill \break
        \hfill \break
        \large
        \@title \\
        \hfill \break
        \hfill \break
        %\LARGE
        %\@author \\
        \large
        %Numer~albumu \@album \\
        \hfill \break
        \hfill \break
        %promotor \\
        %\@promotor \\
        \vfill
        Buenos Aires \@date
    \end{center}
    
    
    \newpage
    
    \pagenumbering{arabic}
    \setcounter{page}{3}
}


%--------------------------------
% Pusta strona - makro
%--------------------------------
\newcommand{\blankpage}{
    \newpage
    \thispagestyle{empty}
    \ 
    \newpage
}

%--------------------------------
% Wykaz symboli i skrótów
%--------------------------------

\newenvironment{acronym}{
    \section*{Wykaz symboli i skrótów}
    \small
    \begin{enumerate}[leftmargin=14.5pt]
}{
    \end{enumerate}
    \normalsize
}

%--------------------------------
% Spis załączników
%--------------------------------
\renewenvironment{appendix}{
    \section*{Spis załączników}
    \small
    \begin{enumerate}[leftmargin=14.5pt]
}{
    \end{enumerate}
    \normalsize
}

%--------------------------------
% Spis generyczny
%--------------------------------
\newenvironment{spis}[1]{
    \section*{Spis #1}
    \small
    \begin{enumerate}[leftmargin=14.5pt]
}{
    \end{enumerate}
    \normalsize
}

% don't show number in front of sections
\makeatletter
\def\@seccntformat#1{%
  \expandafter\ifx\csname c@#1\endcsname\c@section\else
  \csname the#1\endcsname\quad
  \fi}
\makeatother

