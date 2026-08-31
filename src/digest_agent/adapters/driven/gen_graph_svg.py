"""Generate a polished SVG diagram of the digest-agent LangGraph pipeline.

Writes ``graph.svg`` into the repo root. Nodes are grouped by pipeline
stage with distinct colors, plus a legend and highlighted loops.
"""
import os

NODES = [
    ("fetch",    "fetch_github (activity + diffs)", 40, 30, "data"),
    ("select",   "select_and_test_model (model selection)", 40, 110, "data"),
    ("summarize", "summarize (LLM narrative)", 40, 210, "gen"),
    ("r0", "reviewer: detail", 250, 80, "review"),
    ("r1", "reviewer: accuracy", 250, 160, "review"),
    ("r2", "reviewer: tone", 250, 240, "review"),
    ("agg",  "aggregate_review (consensus)", 250, 360, "review"),
    ("refine", "refine", 480, 200, "review"),
    ("final", "finalize (prepend deterministic)", 250, 470, "final"),
    ("render", "render_pdf", 480, 430, "final"),
    ("notify", "send_notification", 480, 520, "final"),
    ("fail",   "handle_failure", 20, 330, "fail"),
]

EDGES = [
    ("fetch", "summarize", "", False),
    ("select", "summarize", "pass", False),
    ("select", "select", "continue", True),
    ("select", "fail", "fail", True),
    ("summarize", "r0", "", False),
    ("summarize", "r1", "", False),
    ("summarize", "r2", "", False),
    ("r0", "agg", "", False),
    ("r1", "agg", "", False),
    ("r2", "agg", "", False),
    ("agg", "final", "accept", False),
    ("agg", "refine", "refine", True),
    ("refine", "r0", "", True),
    ("refine", "r1", "", True),
    ("refine", "r2", "", True),
    ("final", "render", "", False),
    ("render", "notify", "", False),
]

STAGE = {
    "data":   ("#e1effe", "#1d4ed8"),
    "gen":    ("#e7f6ec", "#15803d"),
    "review": ("#fff4e5", "#b4530a"),
    "final":  ("#f3e8fd", "#7e22ce"),
    "fail":   ("#fde2e2", "#b91c1c"),
}

BW = 190
BH = 28


def _esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_svg(path):
    pos = {n[0]: (n[2], n[3]) for n in NODES}
    W, H = 720, 580
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>"]
    out.append(
        "<style> text{font-family:Inter,Helvetica,Arial,sans-serif;fill:#111827;font-size:11px} "
        ".sub{font-size:9px;fill:#374151} "
        ".lbl{font-size:9px;fill:#6b7280;font-style:italic} "
        ".edge{stroke:#9ca3af;stroke-width:1.3;fill:none} "
        ".cycle{stroke:#dc2626;stroke-width:1.6;stroke-dasharray:5,3} "
        ".legend{font-size:10px;fill:#111827} "
        "</style>")

    # edges
    for a, b, label, is_cycle in EDGES:
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        cls = "edge cycle" if is_cycle else "edge"
        xa, ya = x1 + BW, y1 + BH // 2
        xb, yb = x2, y2 + BH // 2
        if a == b:  # self loop on select
            out.append(
                f"<path class='cycle' d='M {xa} {ya} C {xa+40} {ya-30}, {xa+40} {ya-30}, {xa} {ya-28}' />")
            continue
        out.append(f"<line class='{cls}' x1='{xa}' y1='{ya}' x2='{xb}' y2='{yb}' />")
        if label:
            mx, my = (xa + xb) // 2, (ya + yb) // 2 - 4
            out.append(f"<text class='lbl' x='{mx}' y='{my}' text-anchor='middle'>{label}</text>")

    # nodes
    for nid, label, x, y, stage in NODES:
        bg, bd = STAGE[stage]
        out.append(f"<rect x='{x}' y='{y}' width='{BW}' height='{BH}' rx='9' fill='{bg}' "
                   f"stroke='{bd}' stroke-width='1.5'/>")
        out.append(f"<text x='{x + BW // 2}' y='{y + 18}' text-anchor='middle'>{_esc(label)}</text>")

    # legend
    lx, ly = 560, 30
    out.append("<g class='legend'>")
    for i, (name, (bg, bd)) in enumerate(STAGE.items()):
        yy = ly + i * 22
        out.append(f"<rect x='{lx}' y='{yy}' width='20' height='15' rx='4' fill='{bg}' stroke='{bd}'/>")
        out.append(f"<text x='{lx + 30}' y='{yy + 13}'>{name}</text>")
    out.append("<text x='560' y='160' class='lbl'>-- dashed = loop</text>")
    out.append("</g>")

    out.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    return path


def main():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "graph.svg")
    render_svg(root)
    print("wrote", root)


if __name__ == "__main__":
    main()