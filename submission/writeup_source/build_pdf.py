"""Build the write-up PDF from writeup.md.

  .venv\\Scripts\\python submission\\writeup_source\\build_pdf.py

1. Draws the two diagrams (figures/nature_cnn.png, figures/update_flow.png) with matplotlib.
2. Converts writeup.md (a small Markdown subset, see parse()) to
   ../actual_submission/CS4287-Assign2_Atari-ID1_ID2_ID3.pdf with reportlab
   (pip install reportlab into the .venv). Fonts: DejaVu, shipped with matplotlib.
The other figures in figures/ are copies of results/ and the experiment outputs.

Markdown subset: front matter (--- key: value ---), # / ## headings, paragraphs, "- " bullets,
"1. " numbered items, | tables |, ``` code blocks ```, :::note blocks :::, images
![caption](a.png) or ![caption](a.png|b.png) side by side, with optional {width=0.8}.
Inline: **bold**, *italic*, `code`; [PLACEHOLDER ...] / [RESULTS PENDING ...] are highlighted.
"""
import os, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_PDF = HERE.parent / "actual_submission" / "CS4287-Assign2_Atari-ID1_ID2_ID3.pdf"
FIG = HERE / "figures"
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------- diagrams
def box(ax, x, y, w, h, text, fc, fs=9, ec="#333333", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)

def arrow(ax, x0, y0, x1, y1, text=None, style="-|>", ls="-", color="#333333", fs=8, off=(0, 0.12)):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, lw=1.3, color=color, linestyle=ls, shrinkA=0, shrinkB=0))
    if text:
        ax.text((x0 + x1) / 2 + off[0], (y0 + y1) / 2 + off[1], text, ha="center", va="bottom",
                fontsize=fs, color=color)

def draw_nature_cnn(path):
    fig, ax = plt.subplots(figsize=(11, 3.3)); ax.set_xlim(0, 11); ax.set_ylim(0, 3.3); ax.axis("off")
    layers = [("input\n4 × 84 × 84\n(4 stacked\ngreyscale frames)", "#e8eef7", 1.45),
              ("conv 1\n32 filters 8×8\nstride 4, ReLU\n→ 32 × 20 × 20", "#cfe0f5", 1.45),
              ("conv 2\n64 filters 4×4\nstride 2, ReLU\n→ 64 × 9 × 9", "#b5d0f0", 1.45),
              ("conv 3\n64 filters 3×3\nstride 1, ReLU\n→ 64 × 7 × 7", "#9cc0ea", 1.45),
              ("flatten\n3,136", "#f2f2f2", 0.95),
              ("dense\n512 units\nReLU", "#f9dcb8", 1.1),
              ("output\n6 Q-values\n(one per\naction)", "#f6c28b", 1.15)]
    x = 0.1
    for i, (t, c, w) in enumerate(layers):
        box(ax, x, 0.75, w, 1.8, t, c, fs=8.8, bold=(i in (0, 6)))
        if i < len(layers) - 1:
            arrow(ax, x + w + 0.02, 1.65, x + w + 0.2, 1.65)
        x += w + 0.22
    ax.text(5.5, 3.05, "Nature DQN network (config A): about 1.69 million parameters",
            ha="center", fontsize=11, fontweight="bold")
    ax.text(5.5, 0.3, "pixels are scaled to [0, 1] inside the network;  action = argmax of the 6 outputs "
            "(or a random action with probability ε)", ha="center", fontsize=8.5, color="#444444")
    fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)

def draw_update_flow(path):
    fig, ax = plt.subplots(figsize=(11, 4.6)); ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.axis("off")
    box(ax, 0.1, 1.8, 1.7, 1.2, "replay buffer\n200,000\ntransitions", "#eeeeee", bold=True)
    box(ax, 2.3, 1.8, 1.6, 1.2, "random batch\nof 32:\n(s, a, r, s′, d)", "#f5f5f5")
    arrow(ax, 1.8, 2.4, 2.3, 2.4)
    box(ax, 4.5, 3.05, 2.1, 1.1, "ONLINE network θ\nQ(s, a; θ)", "#cfe0f5", bold=True)
    box(ax, 4.5, 0.55, 2.1, 1.1, "TARGET network θ⁻\nQ(s′, · ; θ⁻)", "#f9dcb8", bold=True)
    arrow(ax, 3.9, 2.7, 4.5, 3.5, "s, a", off=(-0.15, 0.05))
    arrow(ax, 3.9, 2.1, 4.5, 1.2, "r, s′, d", off=(-0.2, -0.3))
    box(ax, 7.2, 0.55, 2.2, 1.1, "target  y = r + γ(1−d)·V(s′)\n(no gradient)", "#fdf0dd", fs=8.5)
    arrow(ax, 6.6, 1.1, 7.2, 1.1)
    box(ax, 7.2, 3.05, 2.2, 1.1, "TD error δ = y − Q(s, a; θ)\nHuber loss L(δ)", "#e5f2e5", fs=8.5)
    arrow(ax, 6.6, 3.6, 7.2, 3.6)
    arrow(ax, 8.3, 1.65, 8.3, 3.05)
    ax.text(8.18, 2.75, "y", fontsize=9, ha="right")
    box(ax, 9.7, 3.05, 1.2, 1.1, "backprop\n∂L/∂θ\nclip ‖g‖ ≤ 10", "#e5f2e5", fs=8)
    arrow(ax, 9.4, 3.6, 9.7, 3.6)
    box(ax, 9.7, 1.75, 1.2, 0.95, "Adam step\nupdates θ\n(lr 1e-4)", "#cfe0f5", fs=8)
    arrow(ax, 10.3, 3.05, 10.3, 2.7)
    ax.annotate("", xy=(6.45, 3.05), xytext=(9.7, 2.0),
                arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#1f5fa8", connectionstyle="arc3,rad=-0.15"))
    ax.text(9.0, 1.95, "new weights", fontsize=8, color="#1f5fa8", ha="center", va="top")
    arrow(ax, 5.25, 3.05, 5.25, 1.65, None, ls="--", color="#b25d00")
    ax.text(5.35, 2.25, "copy θ → θ⁻\nevery 1,000\ndecisions", fontsize=8, color="#b25d00", va="center")
    ax.text(0.1, 1.05, "Double DQN only: the online network\nalso reads s′ and picks the argmax\naction; "
            "the target network evaluates it.", fontsize=7.5, color="#1f5fa8", va="center")
    ax.text(5.5, 4.45, "One Q-learning update (every 4 decisions)", ha="center", fontsize=11, fontweight="bold")
    ax.text(0.1, 0.15, "V(s′) = max over a′ of Q(s′, a′; θ⁻) for DQN;  Q(s′, argmax over a′ of Q(s′, a′; θ); θ⁻) "
            "for Double DQN.   d = 1 if the game ended.", fontsize=8.3, color="#444444")
    fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- PDF
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate,
                                Paragraph, Preformatted, Spacer, Table, TableStyle, ListFlowable, ListItem)

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, file in (("Sans", "DejaVuSans.ttf"), ("Sans-Bold", "DejaVuSans-Bold.ttf"),
                   ("Sans-Italic", "DejaVuSans-Oblique.ttf"), ("Sans-BoldItalic", "DejaVuSans-BoldOblique.ttf"),
                   ("Mono", "DejaVuSansMono.ttf")):
    pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / file)))
addMapping("Sans", 0, 0, "Sans"); addMapping("Sans", 1, 0, "Sans-Bold")
addMapping("Sans", 0, 1, "Sans-Italic"); addMapping("Sans", 1, 1, "Sans-BoldItalic")

ACCENT = colors.HexColor("#1f4e8c")
S = {
    "body": ParagraphStyle("body", fontName="Sans", fontSize=9.3, leading=12.7, spaceAfter=5),
    "h1": ParagraphStyle("h1", fontName="Sans-Bold", fontSize=14.5, leading=18, spaceBefore=12, spaceAfter=6,
                         textColor=ACCENT),
    "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=11.3, leading=14.5, spaceBefore=8, spaceAfter=4,
                         textColor=ACCENT),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=8.0, leading=10.0),
    "cellh": ParagraphStyle("cellh", fontName="Sans-Bold", fontSize=8.0, leading=10.0, textColor=colors.white),
    "cap": ParagraphStyle("cap", fontName="Sans-Italic", fontSize=8.3, leading=10.8, textColor=colors.HexColor("#444444"),
                          alignment=TA_CENTER, spaceBefore=3, spaceAfter=9),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=7.9, leading=10.6, leftIndent=8,
                           backColor=colors.HexColor("#f4f6f9"), borderPadding=(5, 5, 5, 5), spaceBefore=4,
                           spaceAfter=9),
    "ref": ParagraphStyle("ref", fontName="Sans", fontSize=8.4, leading=11, leftIndent=16, firstLineIndent=-16,
                          spaceAfter=4),
}

def inline(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"`([^`]+)`", r'<font name="Mono" size="8.6">\1</font>', t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"\[(PLACEHOLDER[^\]]*|RESULTS PENDING[^\]]*)\]",
               r'<font color="#b03a00"><b>[\1]</b></font>', t)
    t = re.sub(r"(https?://[^\s,)]+)", r'<link href="\1" color="#1f4e8c">\1</link>', t)
    return t

def para(text, style="body"):
    return Paragraph(inline(text), S[style])

def table(rows, width):
    cells = [[r.strip() for r in re.split(r"(?<!\\)\|", row.strip().strip("|"))] for row in rows]
    head, body = cells[0], [c for c in cells[2:]]
    n = len(head)
    lens = [max(len(r[i]) if i < len(r) else 0 for r in cells) for i in range(n)]
    lens = [min(max(l, 8), 60) for l in lens]
    widths = [width * l / sum(lens) for l in lens]
    data = [[Paragraph(inline(h), S["cellh"]) for h in head]] + \
           [[Paragraph(inline(c), S["cell"]) for c in r] for r in body]
    tb = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    tb.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5fa")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c5cedb")),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    # Never split a table across pages (e.g. the network-structure table in 3b): if it doesn't fit
    # in the space left, the whole table moves to the next page.
    return [KeepTogether([tb]), Spacer(1, 8)]

def image(spec, width):
    m = re.match(r"!\[(.*)\]\(([^)]*)\)(\{width=([\d.]+)\})?", spec)
    caption, paths, frac = m.group(1), m.group(2).split("|"), float(m.group(4) or 1.0)
    w = width * frac / len(paths) - (4 if len(paths) > 1 else 0)
    imgs = []
    for p in paths:
        img = Image(str(HERE / p)); ratio = img.imageHeight / img.imageWidth
        img.drawWidth, img.drawHeight = w, w * ratio
        imgs.append(img)
    obj = imgs[0] if len(imgs) == 1 else Table([imgs], colWidths=[w + 8] * len(imgs))
    return [KeepTogether([obj, Paragraph(inline(caption), S["cap"])])]

def bullets(items, numbered=False, start=1):
    """items: list of (text, [sub-bullet texts])."""
    out = []
    for text, subs in items:
        content = [para(text)]
        if subs:
            content.append(ListFlowable([ListItem(para(x), leftIndent=10) for x in subs], bulletType="bullet",
                                        leftIndent=12, bulletFontName="Sans", bulletFontSize=7.5))
        out.append(ListItem(content, leftIndent=12))
    kw = dict(bulletType="1", start=str(start)) if numbered else dict(bulletType="bullet", start="•")
    return [ListFlowable(out, leftIndent=14, bulletFontName="Sans", bulletFontSize=9 if numbered else 8.5, **kw),
            Spacer(1, 3)]

def parse(md, width):
    meta, story = {}, []
    lines = md.splitlines()
    if lines and lines[0].strip() == "---":
        end = lines.index("---", 1)
        for l in lines[1:end]:
            k, _, v = l.partition(":"); meta[k.strip()] = v.strip()
        lines = lines[end + 1:]
    i, buf = 0, []
    def flush():
        if buf:
            story.append(para(" ".join(buf))); buf.clear()
    while i < len(lines):
        l = lines[i]; s = l.strip()
        if not s:
            flush(); i += 1; continue
        if s.startswith("```"):
            flush(); j = i + 1
            while not lines[j].strip().startswith("```"):
                j += 1
            story.append(Preformatted("\n".join(lines[i + 1:j]), S["code"])); i = j + 1; continue
        if s.startswith(":::"):
            flush(); j = i + 1
            while lines[j].strip() != ":::":
                j += 1
            inner = parse("\n".join(lines[i + 1:j]), width - 16)[1]
            tb = Table([[inner]], colWidths=[width])
            tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff6e5")),
                                    ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#e0a040")),
                                    ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                                    ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
            story += [tb, Spacer(1, 8)]; i = j + 1; continue
        if s.startswith("# "):
            flush(); story.append(Paragraph(inline(s[2:]), S["h1"])); i += 1; continue
        if s.startswith("## "):
            flush(); story.append(Paragraph(inline(s[3:]), S["h2"])); i += 1; continue
        if s.startswith("!["):
            flush(); story += image(s, width); i += 1; continue
        if s.startswith("|"):
            flush(); j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            story += table(lines[i:j], width); i = j; continue
        if s.startswith("- ") or re.match(r"\d+\. ", s):
            flush(); numbered = not s.startswith("- "); items = []
            start = int(s.split(".")[0]) if numbered else 1
            while i < len(lines) and lines[i].strip():
                raw, t = lines[i], lines[i].strip()
                if raw.startswith("   ") and t.startswith("- ") and items:      # indented sub-bullet
                    items[-1][1].append(t[2:])
                elif numbered and re.match(r"\d+\. ", t):
                    items.append((t.split(" ", 1)[1], []))
                elif not numbered and t.startswith("- "):
                    items.append((t[2:], []))
                else:
                    break
                i += 1
            story += bullets(items, numbered, start); continue
        if re.match(r"\[\d+\] ", s):
            flush(); story.append(para(s, "ref")); i += 1; continue
        buf.append(s); i += 1
    flush()
    return meta, story

def title_page(meta, width):
    out = [Spacer(1, 45 * mm)]
    out.append(Paragraph(inline(meta["title"]), ParagraphStyle("t", fontName="Sans-Bold", fontSize=22, leading=28,
                                                                 alignment=TA_CENTER, textColor=ACCENT)))
    out.append(Spacer(1, 6))
    out.append(Paragraph(inline(meta.get("subtitle", "")), ParagraphStyle("st", fontName="Sans", fontSize=12.5,
                                                                            leading=17, alignment=TA_CENTER)))
    out.append(Spacer(1, 16 * mm))
    banner = Table([[Paragraph(inline(meta["banner"]), ParagraphStyle("b", fontName="Sans-Bold", fontSize=15,
                                                                       leading=19, alignment=TA_CENTER,
                                                                       textColor=colors.HexColor("#9a1c1c")))]],
                   colWidths=[width * 0.8])
    banner.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 2, colors.HexColor("#9a1c1c")),
                                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fdecec")),
                                ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    out += [banner, Spacer(1, 18 * mm)]
    sty = ParagraphStyle("m", fontName="Sans", fontSize=11, leading=17, alignment=TA_CENTER)
    out.append(Paragraph("<b>Team</b>", sty))
    for member in meta["team"].split("·"):
        out.append(Paragraph(inline(member.strip()), sty))
    out += [Spacer(1, 12 * mm), Paragraph(inline(meta["module"]), sty), Paragraph(inline(meta["date"]), sty),
            PageBreak()]
    return out

def on_page(canvas, doc):
    if doc.page == 1:
        return
    canvas.saveState(); canvas.setFont("Sans", 7.8); canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(20 * mm, 12 * mm, "CS4287 Assignment 2: DQN for Atari Pong  |  DRAFT: results pending main experiment")
    canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, f"page {doc.page}")
    canvas.restoreState()

def main():
    FIG.mkdir(exist_ok=True)
    draw_nature_cnn(FIG / "nature_cnn.png"); draw_update_flow(FIG / "update_flow.png")
    doc = BaseDocTemplate(str(OUT_PDF), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=16 * mm,
                          bottomMargin=18 * mm, title="CS4287 Assignment 2: Deep Q-Networks for Atari Pong (draft)",
                          author="CS4287 team [PLACEHOLDER]")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=on_page)])
    # Record the page(s) on which every table is drawn, so a split table is visible in the build output.
    placed = []
    sections = []
    def after(flowable):
        if isinstance(flowable, Paragraph) and flowable.style is S["h1"]:
            sections.append((flowable.getPlainText(), doc.page))
        if isinstance(flowable, Table) and flowable._cellvalues:
            cell = flowable._cellvalues[0][0]
            cell = cell[0] if isinstance(cell, (list, tuple)) and cell else cell
            if hasattr(cell, "getPlainText") and getattr(cell, "style", None) is S["cellh"]:   # data tables only
                placed.append((cell.getPlainText(), len(flowable._cellvalues), doc.page))
    doc.afterFlowable = after
    meta, story = parse((HERE / "writeup.md").read_text(encoding="utf-8"), doc.width)
    doc.build(title_page(meta, doc.width) + story)
    print(f"wrote {OUT_PDF}  ({doc.page} pages)")
    print("sections:", ", ".join(f"{t.split('.')[0]}: p{p}" for t, p in sections))
    print("tables (first header cell, rows drawn, page):")
    for head, rows, page in placed:
        print(f"  page {page:2d}  {rows:2d} rows  '{head}'")

if __name__ == "__main__":
    main()
