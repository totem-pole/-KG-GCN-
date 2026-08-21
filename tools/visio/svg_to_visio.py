#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert the editable SVG framework into native Microsoft Visio shapes.

Windows only. Requires Microsoft Visio desktop and pywin32.
Usage:
  python tools/visio/svg_to_visio.py input.svg output.vsdx
"""
from __future__ import annotations
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import win32com.client

SVG_NS = "{http://www.w3.org/2000/svg}"
PX_PER_IN = 100.0


def num(v, d=0.0):
    if not v:
        return d
    m = re.search(r"[-+]?\d*\.?\d+", str(v))
    return float(m.group()) if m else d


def rgb(c, fallback="#000000"):
    c = (c or fallback).strip()
    if c in ("none", "transparent"):
        return "RGB(255,255,255)"
    if c.startswith("#"):
        h = c[1:]
        if len(h) == 3:
            h = "".join(x * 2 for x in h)
        if len(h) >= 6:
            return f"RGB({int(h[:2],16)},{int(h[2:4],16)},{int(h[4:6],16)})"
    return rgb(fallback)


def style(el):
    s = {}
    for p in el.attrib.get("style", "").split(";"):
        if ":" in p:
            k, v = p.split(":", 1)
            s[k.strip()] = v.strip()
    for k in ["fill", "stroke", "stroke-width", "stroke-dasharray",
              "font-size", "font-weight", "text-anchor", "font-style"]:
        if k in el.attrib:
            s[k] = el.attrib[k]
    return s


def set_shape(s, st, rounding=0.0):
    fill = st.get("fill", "#FFFFFF")
    stroke = st.get("stroke", "#000000")
    try:
        if fill in ("none", "transparent"):
            s.CellsU("FillPattern").FormulaU = "0"
        else:
            s.CellsU("FillPattern").FormulaU = "1"
            s.CellsU("FillForegnd").FormulaU = rgb(fill, "#FFFFFF")
    except Exception:
        pass
    try:
        if stroke in ("none", "transparent"):
            s.CellsU("LinePattern").FormulaU = "0"
        else:
            s.CellsU("LinePattern").FormulaU = "1"
            s.CellsU("LineColor").FormulaU = rgb(stroke)
            s.CellsU("LineWeight").FormulaU = f"{max(.3, num(st.get('stroke-width'),1)*.75):.3f} pt"
        if st.get("stroke-dasharray"):
            s.CellsU("LinePattern").FormulaU = "9"
        if rounding:
            s.CellsU("Rounding").FormulaU = f"{rounding/PX_PER_IN:.4f} in"
    except Exception:
        pass


def set_text(s, txt, st):
    s.Text = txt
    fs = num(st.get("font-size"), 12)
    try:
        s.CellsU("Char.Size").FormulaU = f"{fs} pt"
        fw = st.get("font-weight", "400")
        italic = st.get("font-style", "normal") == "italic"
        bold = fw == "bold" or num(fw, 400) >= 600
        s.CellsU("Char.Style").FormulaU = "3" if bold and italic else ("1" if bold else ("2" if italic else "0"))
        s.CellsU("Char.Color").FormulaU = rgb(st.get("fill", "#111111"), "#111111")
        anchor = st.get("text-anchor", "start")
        s.CellsU("Para.HorzAlign").FormulaU = "1" if anchor == "middle" else ("2" if anchor == "end" else "0")
        s.CellsU("VerticalAlign").FormulaU = "1"
        s.CellsU("LinePattern").FormulaU = "0"
        s.CellsU("FillPattern").FormulaU = "0"
    except Exception:
        pass


def text_value(el):
    ts = el.findall(f"{SVG_NS}tspan")
    if ts:
        return "\n".join("".join(t.itertext()).strip() for t in ts if "".join(t.itertext()).strip())
    return "".join(el.itertext()).strip()


def xy(x, y, ph):
    return x/PX_PER_IN, (ph-y)/PX_PER_IN


def box(x, y, w, h, ph):
    return x/PX_PER_IN, (ph-(y+h))/PX_PER_IN, (x+w)/PX_PER_IN, (ph-y)/PX_PER_IN


def convert(svg_file: Path, out_file: Path):
    root = ET.parse(svg_file).getroot()
    vb = [num(v) for v in root.attrib.get("viewBox", "0 0 1672 941").replace(",", " ").split()]
    pw, ph = vb[2], vb[3]

    app = win32com.client.DispatchEx("Visio.Application")
    app.Visible = True
    doc = app.Documents.Add("")
    page = doc.Pages.Item(1)
    page.Name = "KG-GCN总体框架"
    page.PageSheet.CellsU("PageWidth").FormulaU = f"{pw/PX_PER_IN:.4f} in"
    page.PageSheet.CellsU("PageHeight").FormulaU = f"{ph/PX_PER_IN:.4f} in"

    for el in list(root):
        tag = el.tag.split("}")[-1]
        if tag == "defs":
            continue
        st = style(el)
        try:
            if tag == "rect":
                x,y,w,h = [num(el.attrib.get(k)) for k in ("x","y","width","height")]
                s = page.DrawRectangle(*box(x,y,w,h,ph))
                set_shape(s, st, num(el.attrib.get("rx")))
            elif tag in ("circle", "ellipse"):
                cx, cy = num(el.attrib.get("cx")), num(el.attrib.get("cy"))
                if tag == "circle":
                    rx = ry = num(el.attrib.get("r"))
                else:
                    rx, ry = num(el.attrib.get("rx")), num(el.attrib.get("ry"))
                s = page.DrawOval(*box(cx-rx, cy-ry, 2*rx, 2*ry, ph))
                set_shape(s, st)
            elif tag == "line":
                x1,y1 = xy(num(el.attrib.get("x1")), num(el.attrib.get("y1")), ph)
                x2,y2 = xy(num(el.attrib.get("x2")), num(el.attrib.get("y2")), ph)
                s = page.DrawLine(x1,y1,x2,y2)
                set_shape(s, {**st, "fill":"none"})
                if "marker-end" in el.attrib:
                    s.CellsU("EndArrow").FormulaU = "13"
                if "marker-start" in el.attrib:
                    s.CellsU("BeginArrow").FormulaU = "13"
            elif tag == "polyline":
                pts = []
                for p in el.attrib.get("points", "").split():
                    if "," in p:
                        a,b = p.split(",",1)
                        pts.append((num(a),num(b)))
                for i in range(len(pts)-1):
                    x1,y1 = xy(*pts[i], ph)
                    x2,y2 = xy(*pts[i+1], ph)
                    s = page.DrawLine(x1,y1,x2,y2)
                    set_shape(s, {**st, "fill":"none"})
                    if i == len(pts)-2 and "marker-end" in el.attrib:
                        s.CellsU("EndArrow").FormulaU = "13"
                    if i == 0 and "marker-start" in el.attrib:
                        s.CellsU("BeginArrow").FormulaU = "13"
            elif tag == "text":
                txt = text_value(el)
                if not txt:
                    continue
                x,y = num(el.attrib.get("x")), num(el.attrib.get("y"))
                fs = num(st.get("font-size"),12)
                lines = txt.splitlines() or [txt]
                w = max(35, min(520, max(len(t) for t in lines)*fs*.95))
                h = max(fs*1.5, len(lines)*fs*1.45)
                anchor = st.get("text-anchor", "start")
                x0 = x-w/2 if anchor == "middle" else (x-w if anchor == "end" else x)
                s = page.DrawRectangle(*box(x0,y-fs*1.1,w,h,ph))
                set_text(s,txt,st)
            elif tag == "path":
                # Complex SVG paths here are decorative icons only. Core diagram geometry/text remains native.
                continue
        except Exception as exc:
            print(f"WARN {tag}: {exc}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    doc.SaveAs(str(out_file.resolve()))
    print(out_file.resolve())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: svg_to_visio.py input.svg output.vsdx")
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
