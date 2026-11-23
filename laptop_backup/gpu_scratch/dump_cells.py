import sys, nbformat
nb = nbformat.read(sys.argv[1], as_version=4)
out = open(sys.argv[2], "w", encoding="utf-8")
for i, c in enumerate(nb.cells):
    if c.cell_type == "code":
        out.write(f"\n# ===== CELL {i} ({len(c.source.splitlines())} lines) =====\n{c.source}\n")
    else:
        first = c.source.strip().splitlines()[0] if c.source.strip() else ""
        out.write(f"\n# ----- md {i}: {first[:100]}\n")
