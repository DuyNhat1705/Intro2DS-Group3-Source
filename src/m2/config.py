r"""Constants of the Milestone 2 parser.

Every list that controls cleanup, structure detection or sentence splitting lives here so
that the behaviour of the parser is documented in one place. Changing anything that alters
the output must bump :data:`PARSER_VERSION`, which invalidates the resume checkpoints.
"""

from __future__ import annotations

# Bump on any change that alters parser output (part of the checkpoint signature).
PARSER_VERSION = "m2-1.1.0"

# ---------------------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------------------
MAX_TEX_FILE_BYTES = 5 * 1024 * 1024      # larger .tex files are usually generated data
MAX_BIB_FILE_BYTES = 20 * 1024 * 1024
MAX_INCLUDE_DEPTH = 32
MAX_MACRO_PASSES = 10
MAX_MACRO_GROWTH = 4.0                    # expanded body may grow at most x4 (+64 KB)
MAX_MACRO_GROWTH_ABS = 64 * 1024
MAX_BRACE_SCAN = 20_000                   # bounded search for a closing brace (chars)
MAX_ENV_SCAN = 2_000_000                  # bounded search for \end{env}
PAPER_TIMEOUT_S = 60.0
SENTENCE_CACHE_BYTES = 64 * 1024 * 1024
SHARD_FLUSH_PAPERS = 200
SHARD_FLUSH_BYTES = 64 * 1024 * 1024
WORKER_RSS_LIMIT_MB = 1200
EST_WORKER_RAM_MB = 450

ROOT_PREFERRED_NAMES = ("main", "ms", "paper", "article", "manuscript")

# ---------------------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------------------
SECTION_RANKS = {
    "part": -1,
    "chapter": 0,
    "section": 1,
    "subsection": 2,
    "subsubsection": 3,
    "paragraph": 4,
    "subparagraph": 5,
}

KINDS = (
    "document", "part", "chapter", "section", "subsection", "subsubsection",
    "paragraph", "subparagraph", "abstract", "list", "item", "block",
    "sentence", "formula", "figure", "table",
)

ID_PREFIX = {
    "part": "h", "chapter": "h", "section": "h", "subsection": "h",
    "subsubsection": "h", "paragraph": "h", "subparagraph": "h",
    "abstract": "a", "list": "l", "item": "i", "block": "b",
    "sentence": "s", "formula": "f", "figure": "fig", "table": "tab",
}

LIST_ENVS = frozenset({
    "itemize", "enumerate", "description", "compactitem", "compactenum", "inparaenum",
    "asparaenum", "asparaitem", "enumerate*", "itemize*", "description*", "inlinelist",
    "compactdesc", "tasks", "checklist", "outline",
})

THEOREM_ENVS = frozenset({
    "theorem", "lemma", "corollary", "proposition", "definition", "remark", "example",
    "proof", "claim", "conjecture", "assumption", "observation", "fact", "note",
    "notation", "problem", "question", "exercise", "hypothesis", "property", "condition",
    "axiom", "case", "step", "result", "finding", "insight", "takeaway", "summary",
    "theorem*", "lemma*", "corollary*", "proposition*", "definition*", "remark*",
    "example*", "claim*", "assumption*", "problem*", "observation*",
    "thm", "lem", "cor", "prop", "defn", "dfn", "rem", "rmk", "exm", "ex", "conj",
    "assum", "obs", "proofsketch", "proof*", "solution", "answer",
    "mdframed", "tcolorbox", "framed", "shaded", "quote", "quotation", "displayquote",
    "center", "flushleft", "flushright", "minipage", "boxedminipage", "fbox",
    "promptbox", "prompt", "callout", "highlightbox", "keybox",
})
# Envs whose body is plain prose that is simply inlined (no node of their own).
TRANSPARENT_ENVS = frozenset({
    "center", "flushleft", "flushright", "minipage", "small", "footnotesize",
    "scriptsize", "normalsize", "large", "Large", "spacing", "singlespace",
    "doublespace", "onehalfspace", "sloppypar", "landscape", "adjustbox", "linenomath",
    "multicols", "multicols*", "otherlanguage", "sffamily", "bfseries", "itshape",
    "document", "frontmatter", "mainmatter", "backmatter", "titlepage", "CJK", "CJK*",
    "highlights", "quote", "quotation",
    "displayquote", "mdframed", "tcolorbox", "framed", "shaded", "boxedminipage",
    "fbox", "promptbox", "callout", "highlightbox", "keybox", "spacing*", "savenotes",
    "refsection", "cbunit", "comment*", "addmargin", "addmargin*", "wide",
})
# Containers that become a `block` node. Quote-like / box-like envs are transparent instead.
BLOCK_ENVS = frozenset(e for e in THEOREM_ENVS if e not in TRANSPARENT_ENVS)

ABSTRACT_ENVS = frozenset({"abstract", "abstract*", "Abstract"})
ACK_ENVS = frozenset({"acknowledgments", "acknowledgements", "acks", "acknowledgment",
                      "acknowledgement", "ack", "Acknowledgments", "Acknowledgements"})
APPENDIX_ENVS = frozenset({"appendices", "subappendices"})

FIGURE_ENVS = frozenset({"figure", "figure*", "wrapfigure", "wrapfigure*", "subfigure",
                         "subfigure*", "algorithm", "algorithm*", "SCfigure", "sidewaysfigure",
                         "sidewaysfigure*", "floatingfigure", "marginfigure", "teaserfigure",
                         "listing", "listing*", "chart"})
TABLE_ENVS = frozenset({"table", "table*", "wraptable", "sidewaystable", "sidewaystable*",
                        "subtable", "margintable", "threeparttable", "tablenotes*"})
TABULAR_ENVS = frozenset({"tabular", "tabular*", "tabularx", "tabulary", "longtable",
                          "longtable*", "supertabular", "xtabular", "tabu", "longtabu",
                          "tblr", "longtblr", "talltblr", "NiceTabular", "array"})
VERBATIM_ENVS = ("verbatim", "verbatim*", "Verbatim", "Verbatim*", "BVerbatim", "lstlisting",
                 "minted", "alltt", "comment", "spverbatim", "code", "pycode", "python",
                 "lstinputlisting", "filecontents", "filecontents*", "tcblisting")
# Verbatim-like envs whose body is masked (never parsed); "comment" is removed entirely.
REMOVED_ENVS = frozenset({"comment", "filecontents", "filecontents*", "thebibliography*"})

# Environments dropped from prose with their whole body (non-content).
DROPPED_ENVS = frozenset({
    "tikzpicture", "pgfpicture", "picture", "axis", "forest", "tikzcd", "circuitikz",
    "dot2tex", "pspicture", "overpic", "titlepage*", "thebibliography*", "references",
    "CCSXML", "IEEEbiography", "IEEEbiographynophoto", "biography", "pycode",
    "tocentry", "graphicalabstract", "comment", "IEEEkeywords", "keywords", "keyword",
    "CCSXML*", "teaserfigure_",
})

# Block-math environments (plus $$..$$ and \[..\]). Starred forms included.
BLOCK_MATH_ENVS = frozenset({
    "equation", "equation*", "align", "align*", "gather", "gather*", "multline",
    "multline*", "eqnarray", "eqnarray*", "flalign", "flalign*", "alignat", "alignat*",
    "displaymath", "dmath", "dmath*", "math*", "subequations", "IEEEeqnarray",
    "IEEEeqnarray*", "empheq", "dgroup", "dgroup*", "darray", "split", "xalignat",
    "xalignat*", "xxalignat",
})
# Multi-line block environments: body is wrapped in aligned.
MULTILINE_MATH_ENVS = frozenset({
    "align", "align*", "gather", "gather*", "multline", "multline*", "eqnarray",
    "eqnarray*", "flalign", "flalign*", "alignat", "alignat*", "IEEEeqnarray",
    "IEEEeqnarray*", "xalignat", "xalignat*", "xxalignat", "empheq", "dgroup", "dgroup*",
})

REFERENCE_TITLES = ("references", "bibliography", "reference", "literature cited",
                    "works cited", "bibliographie", "referencias", "literatur",
                    "citations", "reference list", "cited literature")

# ---------------------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------------------
# Commands removed together with their arguments (count = number of brace args, optional
# [..] args are always consumed).
DROP_WITH_ARGS = {
    "label": 1, "vspace": 1, "vspace*": 1, "hspace": 1, "hspace*": 1, "cline": 1,
    "cmidrule": 1, "setlength": 2, "addtolength": 2, "setcounter": 2, "addtocounter": 2,
    "phantom": 1, "hphantom": 1, "vphantom": 1, "includegraphics": 1, "pagestyle": 1,
    "thispagestyle": 1, "bibliographystyle": 1, "bibliography": 1, "addbibresource": 1,
    "nocite": 1, "linespread": 1, "fontsize": 2, "selectfont": 0, "usepackage": 1,
    "RequirePackage": 1, "graphicspath": 1, "definecolor": 3, "color": 1,
    "pagecolor": 1, "rowcolor": 1, "cellcolor": 1, "columncolor": 1, "arrayrulecolor": 1,
    "renewcommand": 2, "newcommand": 2, "providecommand": 2, "vskip": 0, "hskip": 0,
    "enlargethispage": 1, "pagenumbering": 1, "markboth": 2, "markright": 1,
    "addcontentsline": 3, "addtocontents": 2, "index": 1, "glsadd": 1, "hypersetup": 1,
    "captionsetup": 1, "sisetup": 1, "lstset": 1, "tikzset": 1, "pgfplotsset": 1,
    "setcitestyle": 1, "setlist": 1, "newtheorem": 2, "theoremstyle": 1,
    "crefname": 3, "Crefname": 3, "numberwithin": 2, "pdfbookmark": 2,
    "belowcaptionskip": 0, "abovecaptionskip": 0, "rule": 2, "specialrule": 3,
    "tnote": 1, "kern": 0, "tnoteref": 1, "fnref": 1, "corref": 1, "tnotetext": 1,
    "fntext": 1, "cortext": 1, "ead": 1, "thanksref": 1, "footnoteref": 1, "fnmark": 1,
    "printead": 1, "orcidlink": 1, "authorinfo": 1, "hypertarget": 1,
    "keywords_drop": 0, "acmConference": 1, "acmBooktitle": 1, "acmDOI": 1, "acmISBN": 1,
    "acmYear": 1, "acmPrice": 1, "copyrightyear": 1, "setcopyright": 1, "acmJournal": 1,
    "acmVolume": 1, "acmNumber": 1, "acmArticle": 1, "acmMonth": 1, "affiliation": 1,
    "institution": 1, "email": 1, "orcid": 1, "address": 1, "authornote": 1,
    "ccsdesc": 1, "icmlauthor": 2, "icmlaffiliation": 2, "icmlcorrespondingauthor": 2,
    "icmlkeywords": 1, "icmltitlerunning": 1, "icmlsetsymbol": 2, "author": 1,
    "title": 1, "date": 1, "titlerunning": 1, "authorrunning": 1, "institute": 1,
    "shorttitle": 1, "shortauthors": 1, "runningtitle": 1, "runningauthor": 1,
    "affil": 1, "corresp": 1, "thanks": 1, "IEEEauthorblockN": 1, "IEEEauthorblockA": 1,
    "fancyhead": 1, "fancyfoot": 1, "lhead": 1, "rhead": 1, "chead": 1, "lfoot": 1,
    "rfoot": 1, "cfoot": 1, "newpage_drop": 0, "printbibliography": 0, "tableofcontents": 0,
    "listoffigures": 0, "listoftables": 0, "bibliographystyle_drop": 0, "doi_drop": 0,
    "iclrfinalcopy": 0, "newgeometry": 1, "geometry": 1, "titleformat": 5,
    "titlespacing": 4, "allowdisplaybreaks": 0, "DeclareMathOperator": 2,
    "DeclareMathOperator*": 2, "newcolumntype": 2, "newenvironment": 3,
    "renewenvironment": 3, "newif": 0, "def": 0, "let": 0, "input": 1, "include": 1,
    "AtBeginDocument": 1, "AtEndDocument": 1, "pgfplotstableread": 2, "newlength": 1,
    "newcounter": 1, "zref": 1, "noptcrule": 0, "algnewcommand": 2, "SetKwInOut": 2,
    "SetKw": 2, "SetKwFunction": 2, "SetKwProg": 4, "SetAlgoLined": 0, "SetKwInput": 2,
    "DontPrintSemicolon": 0, "LinesNumbered": 0, "newfloat": 3, "floatname": 2,
    "captionof_drop": 0, "subcaptionbox_drop": 0, "affiliations": 1, "authors": 1,
    "correspondingauthor": 1, "keywords": 1, "subjclass": 1, "dedicatory": 1,
    "received": 1, "accepted": 1, "published": 1, "jyear": 1, "jvol": 1, "jnum": 1,
    "editor": 1, "corrauth": 1, "PACS": 1, "pacs": 1, "MSC": 1, "JEL": 1,
    "submitted": 1, "acmSubmissionID": 1, "conferenceinfo": 2, "CopyrightYear": 1,
    "crdata": 1, "numberofauthors": 1, "alignauthor": 0, "and": 0, "AND": 0,
    "paperid": 1, "aistatsauthor": 1, "aistatsaddress": 1, "aistatstitle": 1,
    "vldbTitle": 1, "vldbAuthors": 1, "vldbDOI": 1, "vldbVolume": 1, "vldbNumber": 1,
    "vldbYear": 1, "vldbpagestyle": 0, "pagebreak": 0, "nopagebreak": 0,
    "linebreak": 0, "nolinebreak": 0, "footnotemark": 0, "hfill": 0, "vfill": 0,
    "hrule": 0, "vrule": 0, "hrulefill": 0, "dotfill": 0, "includepdf": 1,
    "input@path": 1, "bibinfo_drop": 0, "endinput": 0, "hdashline": 0, "cdashline": 1,
    "Xhline": 1, "arrayrulewidth": 0, "extrarowheight": 0, "tabcolsep": 0,
    "arraystretch": 0, "baselinestretch": 0, "parindent": 0, "parskip": 0,
    "textwidth": 0, "linewidth": 0, "columnwidth": 0, "textheight": 0,
    "abovedisplayskip": 0, "belowdisplayskip": 0, "printAffiliationsAndNotice": 1,
    "icmlEqualContribution": 0, "twocolumn": 0, "onecolumn": 0, "balance": 0,
    "IEEEoverridecommandlockouts": 0, "IEEEpeerreviewmaketitle": 0, "copyrightnotice": 0,
    "bstctlcite": 1, "nobibliography": 1, "printglossary": 0, "printglossaries": 0,
    "makeatletter": 0, "makeatother": 0, "frenchspacing": 0, "nonfrenchspacing": 0,
}

# Commands dropped without arguments (pure formatting switches).
DROP_SWITCHES = frozenset({
    "centering", "raggedright", "raggedleft", "noindent", "indent", "newpage", "clearpage",
    "cleardoublepage", "toprule", "midrule", "bottomrule", "hline", "maketitle",
    "tiny", "scriptsize", "footnotesize", "small", "normalsize", "large", "Large",
    "LARGE", "huge", "Huge", "bfseries", "itshape", "ttfamily", "sffamily", "rmfamily",
    "mdseries", "upshape", "slshape", "scshape", "normalfont", "em", "bf", "it", "tt",
    "sf", "rm", "sl", "sc", "medskip", "bigskip", "smallskip", "par", "relax",
    "protect", "null", "nobreak", "allowbreak", "break", "displaystyle", "textstyle",
    "scriptstyle", "appendixpage", "addappheadtotoc", "startcontents", "printcontents",
    "FloatBarrier", "flushbottom", "raggedbottom", "sloppy", "fussy", "strut",
    "unskip", "ignorespaces", "leavevmode", "hbox", "vbox", "noalign", "endgraf",
    "xspace", "@", "/", "nointerlineskip", "baselineskip", "lineskip", "selectlanguage",
    "titlepage", "makeatletter", "makeatother", "boldmath", "unboldmath", "onecolumn",
    "twocolumn", "newline", "linebreak", "hfil", "vfil", "hss", "vss", "cr", "crcr",
    "ignorespacesafterend", "global", "long", "outer", "expandafter", "noexpand",
    "smash", "tableofcontents", "printbibliography", "bibliographystyle", "nocite",
    "acknowledgments", "acknowledgements", "IEEEPARstart_drop", "and", "AND",
    "headertitle", "thispagestyle", "hfill", "vfill", "quad", "qquad", "enspace",
    "thinspace", "negthinspace", "enskip", "space", "hrule", "vrule", "footnotemark",
    "arraybackslash", "tabularnewline", "toprule*", "midrule*", "bottomrule*",
    "specialrule", "cmidrule", "morecmidrules", "addlinespace", "endhead", "endfoot",
    "endfirsthead", "endlastfoot", "label", "nonumber", "notag", "appendix",
    "keywordsname", "abstractname", "iclrfinalcopy",
})

# Commands whose content (last brace argument) is kept as plain text.
KEEP_CONTENT = frozenset({
    "textbf", "emph", "textit", "texttt", "underline", "mbox", "textsc", "textsl",
    "textrm", "textsf", "textmd", "textup", "textnormal", "uline", "uwave", "sout",
    "hl", "textcolor", "colorbox", "fcolorbox", "framebox", "makebox", "fbox", "text",
    "textsuperscript", "textsubscript", "ensuremath", "mathrm", "mbox", "hbox",
    "resizebox", "scalebox", "rotatebox", "adjustbox", "parbox", "raisebox",
    "highlight", "added", "replaced", "deleted", "underbar", "bm",
    "boldsymbol", "textup", "foreignlanguage", "lowercase", "uppercase", "MakeUppercase",
    "MakeLowercase", "textls", "so", "st", "ul", "caps", "nolinkurl", "path",
    "smallcaps", "makecell", "shortstack", "thead", "multicolumn", "multirow",
    "cell", "tabularcell", "centerline", "leftline", "rightline", "Centerstack",
    "Longstack", "textrm", "textnormal", "textgreek", "mintinline_drop", "enquote",
    "textquote", "emphasis", "keyword", "name", "term", "acronym",
})

# Commands whose (single) argument is dropped together with the command in prose,
# because it carries metadata, not content.
DROP_ARG_IN_PROSE = frozenset({"footnote", "footnotetext", "thanks", "marginpar", "todo",
                               "label", "index", "comment", "ignore"})

TEXT_SYMBOLS = {
    "%": "%", "&": "&", "_": "_", "#": "#", "$": "$", "{": "{", "}": "}",
    "textbackslash": "\\", "textasciitilde": "~", "textasciicircum": "^",
    "textbar": "|", "textless": "<", "textgreater": ">", "textendash": "–",
    "textemdash": "—", "textbullet": "•", "textdagger": "†", "textdaggerdbl": "‡",
    "textsection": "§", "S": "§", "P": "¶", "textparagraph": "¶", "copyright": "©",
    "textcopyright": "©", "textregistered": "®", "texttrademark": "™", "dag": "†",
    "ddag": "‡", "pounds": "£", "texteuro": "€", "euro": "€", "textdegree": "°",
    "degree": "°", "ldots": "...", "dots": "...", "textellipsis": "...", "LaTeX": "LaTeX",
    "TeX": "TeX", "LaTeXe": "LaTeX2e", "BibTeX": "BibTeX", "ss": "ß", "ae": "æ",
    "AE": "Æ", "oe": "œ", "OE": "Œ", "aa": "å", "AA": "Å", "o": "ø", "O": "Ø",
    "l": "ł", "L": "Ł", "i": "ı", "j": "ȷ", "textquoteleft": "'", "textquoteright": "'",
    "textquotedblleft": '"', "textquotedblright": '"', "quad": " ", "qquad": " ",
    " ": " ", ",": " ", ";": " ", ":": " ", "!": "", "-": "", "slash": "/",
    "textperiodcentered": "·", "checkmark": "✓", "cmark": "✓", "xmark": "✗",
    "times": "×", "textmu": "µ", "textonehalf": "½", "textpm": "±", "pm": "±",
    "textasteriskcentered": "*", "nbsp": " ", "etal": "et al.", "ie": "i.e.",
    "eg": "e.g.", "etc": "etc.", "vs": "vs.", "wrt": "w.r.t.", "cf": "cf.",
    "aka": "a.k.a.", "Eg": "E.g.", "Ie": "I.e.", "iid": "i.i.d.", "st": "s.t.",
    "textunderscore": "_", "guillemotleft": "«", "guillemotright": "»",
    "textexclamdown": "¡", "textquestiondown": "¿", "textsterling": "£",
}

ACCENTS = {
    "'": "́", "`": "̀", "^": "̂", '"': "̈", "~": "̃",
    "=": "̄", ".": "̇", "u": "̆", "v": "̌", "H": "̋",
    "c": "̧", "k": "̨", "d": "̣", "b": "̱", "r": "̊",
    "t": "͡",
}

CITE_COMMANDS = frozenset({
    "cite", "citep", "citet", "citealp", "citealt", "citeauthor", "citeyear",
    "citeyearpar", "parencite", "textcite", "autocite", "footcite", "smartcite",
    "supercite", "citenum", "Citep", "Citet", "Citealp", "Citealt", "Citeauthor",
    "Parencite", "Textcite", "Autocite", "nocite_keep", "fullcite", "footfullcite",
    "citeA", "citeN", "shortcite", "citeasnoun", "possessivecite", "citeyearNP",
    "citeNP", "citeauthorNP", "cites", "parencites", "textcites", "autocites",
    "citetext_keep", "mancite", "citepalias", "citetalias", "Cite", "citeposs",
    "citeal", "citeonline", "upcite", "rcite",
})
REF_COMMANDS = frozenset({
    "ref", "eqref", "autoref", "cref", "Cref", "pageref", "nameref", "vref", "Vref",
    "cpageref", "Autoref", "subref", "labelcref", "crefrange", "Crefrange", "hyperref_ref",
    "fref", "Fref", "prettyref", "thmref", "figref", "tabref", "secref", "Figref",
})

# ---------------------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------------------
ABBREVIATIONS = (
    "i.e.", "e.g.", "et al.", "etc.", "vs.", "cf.", "Fig.", "Figs.", "Eq.", "Eqs.",
    "Sec.", "Secs.", "Tab.", "Tabs.", "Ref.", "Refs.", "No.", "Nos.", "Dr.", "Prof.",
    "Mr.", "Mrs.", "Ms.", "approx.", "resp.", "w.r.t.", "a.k.a.", "i.i.d.", "s.t.",
    "viz.", "Ch.", "Chap.", "Sect.", "Thm.", "Lem.", "Def.", "Prop.", "Cor.", "Alg.",
    "Algo.", "App.", "Appx.", "Apps.", "Ex.", "Exp.", "Eqn.", "Eqns.", "eq.", "eqs.",
    "fig.", "figs.", "sec.", "tab.", "ref.", "refs.", "no.", "vol.", "Vol.", "pp.",
    "p.", "ca.", "esp.", "incl.", "approx.", "Jr.", "Sr.", "St.", "Inc.", "Ltd.",
    "Co.", "Corp.", "Univ.", "Dept.", "dept.", "est.", "max.", "min.", "avg.", "std.",
    "var.", "def.", "w.l.o.g.", "W.l.o.g.", "a.s.", "a.e.", "r.h.s.", "l.h.s.",
    "e.g.,", "i.e.,", "Eg.", "Ie.", "E.g.", "I.e.", "Cf.", "vs", "et~al.", "etal.",
    "Fig.~", "Eq.~", "Sec.~", "Tab.~", "Ref.~", "Table.", "Figure.", "Section.",
    "Def.", "Rem.", "Obs.", "Asm.", "Assump.", "Ass.", "Conj.", "Prob.", "Lemma.",
    "e.~g.", "i.~e.", "U.S.", "U.K.", "U.S.A.", "E.U.", "Ph.D.", "M.Sc.", "B.Sc.",
    "op.", "cit.", "ibid.", "al.", "Mt.", "ft.", "vs.,",
)
# Abbreviations that may genuinely end a sentence (handled by a look-ahead rule).
TERMINAL_ABBREVIATIONS = frozenset({"etc.", "al.", "resp.", "Inc.", "Ltd.", "Co.", "Jr.",
                                    "Sr.", "U.S.", "U.K.", "Ph.D.", "approx.", "ibid.",
                                    "i.i.d.", "a.s.", "a.e."})
