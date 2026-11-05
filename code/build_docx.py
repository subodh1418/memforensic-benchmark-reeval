"""Builds paper/MemTGF_manuscript.docx from the same LaTeX sources as the Springer Nature (sn-jnl) version.

Handles the LaTeX subset used in paper/: sections, paragraphs, itemize/enumerate, inline and display math
(converted to native Word equations through Office's MML2OMML.XSL), figures, tables (incl. \\input bodies),
algorithmic blocks, \\cite (numeric, order of first citation, Springer reference style) and \\ref.
Usage: python build_docx.py
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from latex2mathml.converter import convert as tex2mml
from lxml import etree

import os

PAPER = Path(__file__).resolve().parents[1] / os.environ.get("MEMTGF_PAPER", "paper")
OUT = PAPER / os.environ.get("MEMTGF_DOCX", "MemTGF_manuscript" + ("_v2" if PAPER.name.endswith("v2") else "") + ".docx")
XSL = Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL")
_xslt = etree.XSLT(etree.parse(str(XSL))) if XSL.exists() else None
FONT = "Times New Roman"


# ============================================================== bibliography
def parse_bib(text):
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        start, depth, i = m.end(), 1, m.end()
        while depth and i < len(text):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        body = text[start:i - 1]
        fields, j = {}, 0
        while j < len(body):
            fm = re.compile(r"\s*,?\s*(\w+)\s*=\s*").match(body, j)
            if not fm:
                break
            k, j = fm.group(1).lower(), fm.end()
            if j < len(body) and body[j] == "{":
                d, s = 1, j + 1
                j += 1
                while d and j < len(body):
                    d += {"{": 1, "}": -1}.get(body[j], 0)
                    j += 1
                fields[k] = body[s:j - 1]
            else:
                vm = re.compile(r"[^,]+").match(body, j)
                fields[k] = vm.group(0).strip() if vm else ""
                j = vm.end() if vm else j + 1
        entries[m.group(2)] = {"type": m.group(1).lower(), **fields}
    return entries


def clean_tex(s):
    s = re.sub(r"\{\\v\{?s\}?\}", "š", s)
    s = s.replace("{\\v{s}}", "š").replace("\\&", "&").replace("--", "–").replace("~", " ")
    s = re.sub(r"\\url\{([^}]*)\}", r"\1", s)
    return re.sub(r"[{}]", "", s).strip()


def fmt_author(a):
    a = a.strip()
    if a.startswith("{") and a.endswith("}"):
        return clean_tex(a)
    if "," in a:
        last, first = [x.strip() for x in a.split(",", 1)]
    else:
        parts = a.split()
        last, first = parts[-1], " ".join(parts[:-1])
    initials = "".join(p[0] for p in re.split(r"[\s\-.]+", clean_tex(first)) if p)
    return f"{clean_tex(last)} {initials}".strip()


def fmt_reference(e):
    authors = [fmt_author(a) for a in re.split(r"\s+and\s+", e.get("author", ""))] if e.get("author") else []
    au = ", ".join(authors) if authors else ""
    yr = clean_tex(e.get("year", ""))
    title = clean_tex(e.get("title", ""))
    parts = [f"{au} ({yr}) {title}."]
    if e["type"] == "article":
        vol = clean_tex(e.get("volume", ""))
        iss = clean_tex(e.get("number", ""))
        pg = clean_tex(e.get("pages", ""))
        src = clean_tex(e.get("journal", ""))
        s = src + (f" {vol}" if vol else "") + (f"({iss})" if iss else "") + (f":{pg}" if pg else "")
        parts.append(s + ".")
    elif e["type"] == "inproceedings":
        parts.append(f"In: {clean_tex(e.get('booktitle', ''))}" + (f", pp {clean_tex(e['pages'])}" if e.get("pages") else "") + ".")
    else:
        hp = clean_tex(e.get("howpublished", ""))
        parts.append(hp + (f". {clean_tex(e['note'])}" if e.get("note") else "") + ".")
    if e.get("doi"):
        parts.append(f"https://doi.org/{clean_tex(e['doi']).lower() if e['doi'].lower().startswith('10.') else clean_tex(e['doi'])}")
    return " ".join(p for p in parts if p.strip(". "))


# ============================================================== math
MATH_FIX = [
    (r"\\operatorname\{([^}]*)\}", r"\\mathrm{\1}"), (r"\\(max|min|log|exp|arg)\b", r"\\mathrm{\1}"), (r"\\tfrac", r"\\frac"), (r"\\textstyle", ""), (r"\\mathbb\{1\}", r"\\mathbf{1}"),
    (r"\\lVert", r"\\|"), (r"\\rVert", r"\\|"), (r"\\Vert", r"\\|"), (r"\\Big", ""), (r"\\big", ""), (r"\\qquad", r"\\quad"),
    (r"\\label\{[^}]*\}", ""), (r"\\nonumber", ""),
]


def omml(tex, display=False):
    for a, b in MATH_FIX:
        tex = re.sub(a, b, tex)
    mml = tex2mml(tex.strip(), display="block" if display else "inline")
    if _xslt is None:
        return None
    om = _xslt(etree.fromstring(mml.encode("utf-8"))).getroot()
    return om


# ============================================================== document helpers
class Builder:
    def __init__(self):
        self.doc = Document()
        st = self.doc.styles["Normal"]
        st.font.name, st.font.size = FONT, Pt(11)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        st.paragraph_format.space_after = Pt(4)
        st.paragraph_format.line_spacing = 1.15
        for name, size, ital in (("Heading 1", 13, False), ("Heading 2", 11.5, False), ("Heading 3", 11, True)):
            h = self.doc.styles[name]
            h.font.name, h.font.size, h.font.bold, h.font.italic = FONT, Pt(size), True, ital
            h.font.color.rgb = RGBColor(0, 0, 0)
            rf = h.element.rPr.rFonts
            for att in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
                rf.attrib.pop(qn(att), None)
            for att in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                rf.set(qn(att), FONT)
            h.paragraph_format.space_before, h.paragraph_format.space_after = Pt(12 if name == "Heading 1" else 8), Pt(4)
        sec = self.doc.sections[0]
        sec.page_width, sec.page_height = Cm(21), Cm(29.7)
        for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(sec, side, Cm(2.5))
        self.cite_order, self.labels = [], {}
        self.counters = {"sec": [0, 0, 0], "fig": 0, "tab": 0, "alg": 0, "eq": 0}

    # ---- inline text with math / formatting
    def inline(self, p, text, bold=False, italic=False, size=None):
        text = self.resolve(text)
        tokens = re.split(r"(\$[^$]+\$|\\textbf\{(?:[^{}]|\{[^{}]*\})*\}|\\emph\{(?:[^{}]|\{[^{}]*\})*\}|\\texttt\{(?:[^{}]|\{[^{}]*\})*\}|\\textit\{(?:[^{}]|\{[^{}]*\})*\})", text)
        for t in tokens:
            if not t:
                continue
            if t.startswith("$"):
                om = omml(t[1:-1])
                if om is not None:
                    p._p.append(om)
                else:
                    p.add_run(t[1:-1]).italic = True
                continue
            m = re.match(r"\\(textbf|emph|texttt|textit)\{(.*)\}$", t, re.S)
            if m:
                kind, inner = m.groups()
                self.inline(p, inner, bold=bold or kind == "textbf", italic=italic or kind in ("emph", "textit"), size=size)
                if kind == "texttt":
                    for r in p.runs[-1:]:
                        r.font.name = "Consolas"
                continue
            r = p.add_run(self.plain(t))
            r.bold, r.italic = bold or None, italic or None
            if size:
                r.font.size = Pt(size)

    def plain(self, s):
        s = s.replace("\\{", "\x01").replace("\\}", "\x02")  # protect escaped braces
        rep = [("\\%", "%"), ("\\&", "&"), ("\\_", "_"), ("\\#", "#"), ("~", "\u00a0"), ("---", "—"), ("--", "–"), ("``", "“"), ("''", "”"),
               ("\\ldots", "…"), ("\\dots", "…"), ("\\,", "\u2009"), ("\\ ", " "), ("\\textrightarrow", "→"), ("\\\\", " ")]
        for a, b in rep:
            s = s.replace(a, b)
        s = re.sub(r"\\(?:centering|small|footnotesize|raggedbottom|noindent)\b", "", s)
        s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", "", s)
        return re.sub(r"[{}]", "", re.sub(r"\s+", " ", s)).replace("\x01", "{").replace("\x02", "}")

    def resolve(self, text):
        def cite(m):
            keys = [k.strip() for k in m.group(1).split(",")]
            nums = []
            for k in keys:
                if k not in self.cite_order:
                    self.cite_order.append(k)
                nums.append(self.cite_order.index(k) + 1)
            nums = sorted(nums)
            return "[" + ",".join(map(str, nums)) + "]"

        text = re.sub(r"\\cite\{([^}]*)\}", cite, text)
        text = re.sub(r"(Section|Sect\.|Table|Figure|Fig\.|Algorithm|Eq\.)~?\\ref\{([^}]*)\}",
                      lambda m: f"{m.group(1)}\u00a0{self.labels.get(m.group(2), '?')}", text)
        text = re.sub(r"\\ref\{([^}]*)\}", lambda m: str(self.labels.get(m.group(1), "?")), text)
        return re.sub(r"\\label\{[^}]*\}", "", text)

    def para(self, text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, **kw):
        p = self.doc.add_paragraph()
        p.alignment = align
        self.inline(p, text, **kw)
        return p

    def display_math(self, tex):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        om = omml(tex, display=True)
        if om is not None:
            if om.tag.endswith("}oMathPara"):
                p._p.append(om)
            else:
                para = OxmlElement("m:oMathPara")
                para.append(om)
                p._p.append(para)
        else:
            p.add_run(tex).italic = True

    def heading(self, level, title):
        if getattr(self, "appendix", False) and level == 1:
            self.app_n = getattr(self, "app_n", 0) + 1
            self.doc.add_heading(f"Appendix {chr(64 + self.app_n)} {self.plain(self.resolve(title))}", level=1)
            return
        c = self.counters["sec"]
        c[level - 1] += 1
        for k in range(level, 3):
            c[k] = 0
        num = ".".join(str(x) for x in c[:level])
        self.doc.add_heading(f"{num} {self.plain(self.resolve(title))}", level=level)

    def figure(self, images, caption, width_cm):
        """images: list of (path, fraction of the text width)."""
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for k, (path, frac) in enumerate(images):
            png = PAPER / (str(Path(path).with_suffix(".png")))
            if png.exists():
                p.add_run().add_picture(str(png), width=Cm(width_cm * frac))
            else:
                p.add_run(f"[missing figure: {png.name}]").italic = True
            if k < len(images) - 1:
                p.add_run("  ")
        self.counters["fig"] += 1
        cp = self.doc.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = cp.add_run(f"Fig. {self.counters['fig']} ")
        r.bold, r.font.size = True, Pt(9.5)
        self.inline(cp, caption, size=9.5)

    def table(self, caption, rows):
        self.counters["tab"] += 1
        cp = self.doc.add_paragraph()
        r = cp.add_run(f"Table {self.counters['tab']} ")
        r.bold, r.font.size = True, Pt(9.5)
        self.inline(cp, caption, size=9.5)
        rows = [rw for rw in rows if rw is not None]
        ncol = max(sum(int(re.match(r"\\multicolumn\{(\d+)\}", c.strip()).group(1)) if c.strip().startswith("\\multicolumn") else 1 for c in rw[0]) for rw in rows)
        t = self.doc.add_table(rows=0, cols=ncol)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.style = "Table Grid"
        for cells, rule_above in rows:
            row = t.add_row().cells
            j = 0
            for c in cells:
                c = c.strip()
                span, align = 1, None
                mm = re.match(r"\\multicolumn\{(\d+)\}\{(?:[^{}]|\{[^{}]*\})*\}\{(.*)\}$", c, re.S)  # alignment may hold @{}
                if mm:
                    span, c = int(mm.group(1)), mm.group(2)
                cell = row[j]
                if span > 1 and j + span - 1 < ncol:
                    cell = cell.merge(row[j + span - 1])
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER if j else WD_ALIGN_PARAGRAPH.LEFT
                self.inline(cell.paragraphs[0], c, size=6.5 if ncol >= 9 else (7.5 if ncol >= 7 else 8.5))
                j += span
        self._booktabs(t, rows)
        self.doc.add_paragraph()

    def _booktabs(self, t, rows):
        """Replace the grid with booktabs-style horizontal rules."""
        tblPr = t._tbl.tblPr
        borders = OxmlElement("w:tblBorders")
        for edge, val in (("top", "single"), ("left", "nil"), ("bottom", "single"), ("right", "nil"), ("insideH", "nil"), ("insideV", "nil")):
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), val)
            el.set(qn("w:sz"), "8")
            borders.append(el)
        tblPr.insert_element_before(borders, "w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook", "w:tblCaption", "w:tblDescription")
        for i, (cells, rule_above) in enumerate(rows):
            if rule_above or i == 1:
                for cell in t.rows[i].cells:
                    tcPr = cell._tc.get_or_add_tcPr()
                    b = OxmlElement("w:tcBorders")
                    top = OxmlElement("w:top")
                    top.set(qn("w:val"), "single")
                    top.set(qn("w:sz"), "4")
                    b.append(top)
                    tcPr.insert_element_before(b, "w:shd", "w:noWrap", "w:tcMar", "w:textDirection", "w:tcFitText", "w:vAlign", "w:hideMark")

    def algorithm(self, caption, body):
        self.counters["alg"] += 1
        p = self.doc.add_paragraph()
        r = p.add_run(f"Algorithm {self.counters['alg']} ")
        r.bold = True
        self.inline(p, caption)
        _border(p, "top")
        indent, n = 0, 0
        for line in [l.strip() for l in body.splitlines() if l.strip() and not l.strip().startswith("\\begin") and not l.strip().startswith("\\end")]:
            m = re.match(r"\\(Require|Ensure|State|For|EndFor|If|EndIf|Return|While|EndWhile)\s*(.*)", line)
            if not m:
                continue
            kw, rest = m.groups()
            if kw.startswith("End"):
                indent -= 1
                text = "end " + kw[3:].lower()
            elif kw in ("Require", "Ensure"):
                text = f"\\textbf{{{kw}:}} {rest}"
            elif kw == "Return":
                text = f"\\textbf{{return}} {rest}"
            elif kw in ("For", "If", "While"):
                inner = re.sub(r"^\{(.*)\}$", r"\1", rest)
                text = f"\\textbf{{{kw.lower()}}} {inner} \\textbf{{do}}"
            else:
                text = rest
            n += 1
            q = self.doc.add_paragraph()
            q.paragraph_format.left_indent = Cm(0.6 * max(indent, 0) + 0.2)
            q.paragraph_format.space_after = Pt(0)
            rr = q.add_run(f"{n}: ")
            rr.font.size = Pt(9)
            self.inline(q, text, size=10)
            if kw in ("For", "If", "While"):
                indent += 1
        _border(self.doc.paragraphs[-1], "bottom")


def _border(p, side):
    pPr = p._p.get_or_add_pPr()
    b = OxmlElement("w:pBdr")
    e = OxmlElement(f"w:{side}")
    e.set(qn("w:val"), "single")
    e.set(qn("w:sz"), "6")
    b.append(e)
    pPr.insert_element_before(b, "w:shd", "w:tabs", "w:suppressAutoHyphens", "w:kinsoku", "w:wordWrap", "w:overflowPunct", "w:topLinePunct",
                              "w:autoSpaceDE", "w:autoSpaceDN", "w:bidi", "w:adjustRightInd", "w:snapToGrid", "w:spacing", "w:ind",
                              "w:contextualSpacing", "w:mirrorIndents", "w:suppressOverlap", "w:jc", "w:textDirection", "w:textAlignment",
                              "w:textboxTightWrap", "w:outlineLvl", "w:divId", "w:cnfStyle", "w:rPr", "w:sectPr", "w:pPrChange")


# ============================================================== LaTeX walker
def strip_comments(s):
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in s.splitlines())


def expand_inputs(s):
    def rep(m):
        f = PAPER / (m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex")
        return expand_inputs(strip_comments(f.read_text(encoding="utf-8"))) if f.exists() else ""
    return re.sub(r"\\input\{([^}]*)\}", rep, s)


def table_rows(tab_body):
    rows, rule = [], False
    tab_body = re.sub(r"\\(toprule|bottomrule|botrule)", "", tab_body)
    for chunk in re.split(r"\\\\", tab_body):
        if "\\midrule" in chunk or "\\hline" in chunk:
            rule = True
            chunk = chunk.replace("\\midrule", "").replace("\\hline", "")
        chunk = re.sub(r"\\cmidrule(\([^)]*\))?\{[^}]*\}", "", chunk).strip()
        if not chunk:
            continue
        rows.append((re.split(r"(?<!\\)&", chunk), rule))
        rule = False
    return rows


def prepass(body, b):
    """Assign numbers to labels in document order."""
    sec, fig, tab, alg = [0, 0, 0], 0, 0, 0
    for m in re.finditer(r"\\(section|subsection|subsubsection)\*?\{|\\begin\{(figure\*?|table\*?|algorithm)\}|\\label\{([^}]*)\}", body):
        if m.group(1):
            lvl = {"section": 0, "subsection": 1, "subsubsection": 2}[m.group(1)]
            if body[m.start():m.end()].endswith("*{"):
                continue
            sec[lvl] += 1
            for k in range(lvl + 1, 3):
                sec[k] = 0
            cur = ".".join(str(x) for x in sec[:lvl + 1])
        elif m.group(2):
            env = m.group(2).rstrip("*")
            if env == "figure":
                fig += 1
                cur = fig
            elif env == "table":
                tab += 1
                cur = tab
            else:
                alg += 1
                cur = alg
        elif m.group(3):
            b.labels[m.group(3)] = cur


def walk(body, b):
    pos = 0
    pat = re.compile(r"\\(section|subsection|subsubsection)\*?\{((?:[^{}]|\{[^{}]*\})*)\}|\\paragraph\{([^}]*)\}|\\begin\{(itemize|enumerate|equation\*?|align\*?|figure\*?|table\*?|algorithm)\}")
    buf = []

    def flush():
        txt = "".join(buf)
        buf.clear()
        for para in re.split(r"\n\s*\n", txt):
            para = para.strip()
            if para and b.plain(para).strip():
                b.para(para.replace("\n", " "))

    while True:
        m = pat.search(body, pos)
        if not m:
            buf.append(body[pos:])
            flush()
            return
        buf.append(body[pos:m.start()])
        flush()
        if m.group(1):
            b.heading({"section": 1, "subsection": 2, "subsubsection": 3}[m.group(1)], m.group(2))
            pos = m.end()
            continue
        if m.group(3) is not None:
            end = body.find("\n\n", m.end())
            end = len(body) if end < 0 else end
            p = b.doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            r = p.add_run(m.group(3) + " ")
            r.bold = True
            b.inline(p, body[m.end():end].replace("\n", " "))
            pos = end
            continue
        env = m.group(4)
        close = f"\\end{{{env}}}"
        end = body.find(close, m.end())
        inner = body[m.end():end]
        pos = end + len(close)
        base = env.rstrip("*")
        if base in ("itemize", "enumerate"):
            for k, item in enumerate([x for x in re.split(r"\\item\s*", inner) if x.strip()], 1):
                p = b.doc.add_paragraph(style="List Bullet" if base == "itemize" else "List Number")
                b.inline(p, item.strip().replace("\n", " "))
        elif base in ("equation", "align"):
            for line in re.split(r"\\\\", inner):
                if line.strip():
                    b.display_math(line.replace("&", ""))
        elif base == "figure":
            cap = re.search(r"\\caption\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", inner)
            imgs = re.findall(r"\\includegraphics(?:\[([^\]]*)\])?\{([^}]*)\}", inner)
            width = 16 if env.endswith("*") else 12
            parts = []
            for opt, path in imgs:
                fm = re.search(r"width=([\d.]*)\\(textwidth|linewidth)", opt or "")
                frac = float(fm.group(1) or 1) if fm else 1.0
                parts.append((path, frac * (0.97 if len(imgs) > 1 else 1.0)))
            b.figure(parts, cap.group(1) if cap else "", width)
        elif base == "table":
            cap = re.search(r"\\caption\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", inner)
            tb = re.search(r"\\begin\{tabular\*?\}(?:\{[^{}]*\})?\{(?:[^{}]|\{[^{}]*\})*\}(.*?)\\end\{tabular\*?\}", inner, re.S)
            b.table(cap.group(1) if cap else "", table_rows(tb.group(1)) if tb else [])
        elif base == "algorithm":
            cap = re.search(r"\\caption\{([^}]*)\}", inner)
            b.algorithm(cap.group(1) if cap else "", inner)


def main():
    main_tex = strip_comments((PAPER / "main.tex").read_text(encoding="utf-8"))
    b = Builder()
    title = re.search(r"\\title(?:\[[^\]]*\])?\{([^}]*)\}", main_tex).group(1)
    p = b.doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(title)
    r.bold, r.font.size = True, Pt(16)
    authors = re.findall(r"\\author\*?(?:\[[^\]]*\])?\{\\fnm\{([^}]*)\}\s*\\sur\{([^}]*)\}\}", main_tex)
    b.para(", ".join(f"{f} {s}" for f, s in authors), align=WD_ALIGN_PARAGRAPH.CENTER)
    for aff in re.findall(r"\\affil\*?\[[^\]]*\]\{(.*)\}", main_tex):
        aff = re.sub(r"\\(orgdiv|orgname|city|country)\{([^}]*)\}", r"\2", aff)
        aff = re.sub(r"\\orgaddress\{(.*)\}", r"\1", aff)
        b.para(aff, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, size=9.5)
    em = re.search(r"\\email\{([^}]*)\}", main_tex)
    if em:
        b.para(f"Corresponding author: {em.group(1)}", align=WD_ALIGN_PARAGRAPH.CENTER, size=9.5)

    body = expand_inputs(main_tex.split("\\maketitle", 1)[1].split("\\backmatter", 1)[0])
    pre = expand_inputs(main_tex.split("\\maketitle", 1)[0])
    prepass(body, b)
    ab = re.search(r"\\abstract\{(.*?)\}\s*(?:\\keywords|$)", pre, re.S)
    if ab:
        p = b.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = p.add_run("Abstract ")
        r.bold = True
        b.inline(p, ab.group(1).replace("\n", " ").strip(), size=10)
    kw = re.search(r"\\keywords\{([^}]*)\}", main_tex)
    if kw:
        p = b.doc.add_paragraph()
        r = p.add_run("Keywords ")
        r.bold = True
        b.inline(p, kw.group(1).replace(",", " ·"), size=10)

    back = main_tex.split("\\backmatter", 1)[1].split("\\bibliography", 1)[0]
    app = re.search(r"\\begin\{appendices\}(.*?)\\end\{appendices\}", expand_inputs(back), re.S)
    if app:  # appendix labels: sections lettered, tables/figures continue the body numbering
        n_tab = len(re.findall(r"\\begin\{table\*?\}", body))
        n_fig = len(re.findall(r"\\begin\{figure\*?\}", body))
        sec_n, cur = 0, None
        for m in re.finditer(r"\\section\{|\\begin\{(table|figure)\*?\}|\\label\{([^}]*)\}", app.group(1)):
            if m.group(0).startswith("\\section"):
                sec_n += 1
                cur = chr(64 + sec_n)
            elif m.group(1) == "table":
                n_tab += 1
                cur = n_tab
            elif m.group(1) == "figure":
                n_fig += 1
                cur = n_fig
            else:
                b.labels[m.group(2)] = cur

    walk(body, b)

    ack = re.search(r"\\bmhead\{Acknowledgements\}\s*(.*?)\n\s*\n", back, re.S)
    if ack:
        b.doc.add_heading("Acknowledgements", level=1)
        b.para(ack.group(1).strip())
    dec = re.search(r"\\section\*\{Declarations\}(.*)", back, re.S)
    if dec:
        b.doc.add_heading("Declarations", level=1)
        for item in [x for x in re.split(r"\\item\s*", dec.group(1).split("\\end{itemize}")[0]) if x.strip() and "\\begin" not in x]:
            b.doc.add_paragraph(style="List Bullet")
            b.inline(b.doc.paragraphs[-1], item.strip())

    if app:
        b.appendix = True
        walk(app.group(1), b)

    bib = parse_bib((PAPER / "references.bib").read_text(encoding="utf-8"))
    b.doc.add_heading("References", level=1)
    for n, k in enumerate(b.cite_order, 1):
        p = b.doc.add_paragraph()
        p.paragraph_format.left_indent, p.paragraph_format.first_line_indent = Cm(0.7), Cm(-0.7)
        r = p.add_run(f"{n}. ")
        r.font.size = Pt(9.5)
        rr = p.add_run(fmt_reference(bib[k]) if k in bib else f"[missing reference: {k}]")
        rr.font.size = Pt(9.5)
    zoom = b.doc.settings.element.find(qn("w:zoom"))
    if zoom is not None and zoom.get(qn("w:percent")) is None:
        zoom.set(qn("w:percent"), "100")
    b.doc.save(OUT)
    print(f"wrote {OUT}  ({len(b.cite_order)} references, {b.counters['fig']} figures, {b.counters['tab']} tables)")


if __name__ == "__main__":
    main()
