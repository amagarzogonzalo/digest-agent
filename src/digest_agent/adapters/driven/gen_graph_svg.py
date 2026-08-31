"""Generate an SVG diagram of the digest-agent LangGraph pipeline.

The SVG is written to ``graph.svg`` in the repo root for visual reference.
"""
import os


NODES = [
    # (id, label, x, y)
    ("fetch_github", "fetch_github", 80, 40),
    ("select", "select_and_test_model", 80, 140),
    ("summarize", "summarize", 80, 240),
    ("review_0", "reviewer: detail/code", 300, 180),
    ("review_1", "reviewer: accuracy", 300, 260),
    ("review_2", "reviewer: structure/tone", 300, 340),
    ("aggregate", "aggregate_review", 520, 260),
    ("refine", "refine", 520, 140),
    ("finalize", "finalize (prepend deterministic)", 700, 200),
    ("render", "render_pdf", 700, 300),
    ("notify", "send_notification", 840, 300),
    ("fail", "handle_failure", 300, 360),
]

EDGES = [
    ("fetch_github", "select"),
    ("select", "summarize", "pass"),
    ("select", "select", "continue"),
    ("select", "fail", "fail_exhausted/budget"),
    ("summarize", "review_0"),
    ("summarize", "review_1"),
    ("summarize", "review_2"),
    ("review_0", "aggregate"),
    ("review_1", "aggregate"),
    ("review_2", "aggregate"),
    ("aggregate", "finalize", "accept"),
    ("aggregate", "refine", "refine"),
    ("refine", "review_0"),
    ("refine", "review_1"),
    ("refine", "review_2"),
    ("finalize", "render"),
    ("render", "notify"),
    ("notify", None),
    ("fail", None),
]


def _node_id(nid):
    return "n" + nid.replace("-", "_")


def render_svg(path: str) -> None:
    w, h = 980, 460
    out = ["<svg xmlns='http://www.w3.org/2000/svg' width='%d' height='%d' viewBox='0 0 %d %d'>" % (w, h, w, h)]
    out.append("<style> text{font-family:sans-serif;font-size:11px;fill:#1f2937} .edge{stroke:#9ca3af;stroke-width:1.2;fill:none} "
               " .lbl{font-size:9px;fill:#6b7280} .node{fill:#eef2f8;stroke:#1f3b73;stroke-width:1.5} "
               " .fail{fill:#fde2e2;stroke:#b91c1c} .cycle{stroke:#b45309;stroke-width:1.6} </style>")
    # edges first
    for e in EDGES:
        a, b = e[0], e[1]
        lbl = e[2] if len(e) > 2 else ""
        if b is None:
            continue
        pos = {n[0]: (n[2], n[3]) for n in NODES}
        (x1, y1), (x2, y2) = pos[a], pos[b]
        cls = "edge cycle" if (a == b or (a == "refine" and b in ("review_0", "review_1", "review_2"))) else "edge"
        out.append("<line class='%s' x1='%d' y1='%d' x2='%d' y2='%d' marker-end='url(#a)'/>" % (cls, x1 + 70, y1 + 18, x2, y2 + 2))
        if lbl:
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            out.append("<text class='lbl' x='%d' y='%d'>%s</text>" % (mx, my, lbl))
    # nodes
    for n in NODES:
        nid, label, x, y = n
        cls = "node fail" if nid == "fail" else "node"
        out.append("<rect class='%s' id='%s' x='%d' y='%d' width='140' height='36' rx='8'/>" % (cls, nid, x, y))
        out.append("<text x='%d' y='%d' text-anchor='middle'>%s</text>" % (x + 70, y + 22, label))
    # marker
    out.append("<defs><marker id='a' markerWidth='8' markerHeight='8' refX='7' refY='3' orient='auto'>"
               "<path d='M0,0 L7,3 L0,6 z' fill='#9ca3af'/></marker></defs>")
    out.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "graph.svg")
    render_svg(root)
    print("wrote", root)