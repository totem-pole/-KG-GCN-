#!/usr/bin/env python3
"""Move the shared legend of Figure 2-7 into the inter-row whitespace.

The figure source is data-faithful and should not be redrawn during thesis build.
This patch changes only the four legend SVG elements, avoiding overlap with the
upper-right subplot title. It runs on the disposable build copy.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "figures/ch2/fig2_7_prediction_panels.svg"
text = path.read_text(encoding="utf-8")
old = (
    '<line x1="800" y1="18" x2="835" y2="18" stroke="black" stroke-width="1.5"/>'
    '<text x="845" y="23" font-size="13">DCS实测</text>'
    '<line x1="925" y1="18" x2="960" y2="18" stroke="#777" stroke-width="1.5" stroke-dasharray="6,4"/>'
    '<text x="970" y="23" font-size="13">GRU预测</text>'
)
new = (
    '<line x1="390" y1="349" x2="425" y2="349" stroke="black" stroke-width="1.5"/>'
    '<text x="435" y="354" font-size="13">DCS实测</text>'
    '<line x1="535" y1="349" x2="570" y2="349" stroke="#777" stroke-width="1.5" stroke-dasharray="6,4"/>'
    '<text x="580" y="354" font-size="13">GRU预测</text>'
)
if old not in text:
    raise RuntimeError("Figure 2-7 legend signature not found; source layout may have changed")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("Figure 2-7 legend moved to inter-row whitespace.")
