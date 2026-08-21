#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import win32com.client
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("visio-paper-figures")
APP = DOC = PAGE = None


def _app():
    global APP
    if APP is None:
        APP = win32com.client.DispatchEx("Visio.Application")
        APP.Visible = True
    return APP


def _shape(shape_id: int):
    if PAGE is None:
        raise RuntimeError("No active page")
    return PAGE.Shapes.ItemFromID(int(shape_id))


@mcp.tool()
def create_blank(width_in: float = 16.72, height_in: float = 9.41, page_name: str = "KG-GCN总体框架") -> str:
    global DOC, PAGE
    app = _app()
    DOC = app.Documents.Add("")
    PAGE = DOC.Pages.Item(1)
    PAGE.Name = page_name
    PAGE.PageSheet.CellsU("PageWidth").FormulaU = f"{width_in} in"
    PAGE.PageSheet.CellsU("PageHeight").FormulaU = f"{height_in} in"
    return f"created {page_name}"


@mcp.tool()
def open_vsdx(path: str, page_index: int = 1) -> str:
    global DOC, PAGE
    app = _app()
    DOC = app.Documents.Open(str(Path(path).resolve()))
    PAGE = DOC.Pages.Item(page_index)
    return f"opened {DOC.FullName}; page={PAGE.Name}"


@mcp.tool()
def list_shapes() -> list[dict]:
    if PAGE is None:
        raise RuntimeError("No active page")
    out = []
    for i in range(1, PAGE.Shapes.Count + 1):
        s = PAGE.Shapes.Item(i)
        out.append({
            "id": int(s.ID),
            "name": str(s.NameU),
            "text": str(s.Text or ""),
            "pin_x": float(s.CellsU("PinX").ResultIU),
            "pin_y": float(s.CellsU("PinY").ResultIU),
            "width": float(s.CellsU("Width").ResultIU),
            "height": float(s.CellsU("Height").ResultIU),
        })
    return out


@mcp.tool()
def move_resize(shape_id: int, pin_x: float | None = None, pin_y: float | None = None,
                width: float | None = None, height: float | None = None) -> str:
    s = _shape(shape_id)
    for cell, val in [("PinX", pin_x), ("PinY", pin_y), ("Width", width), ("Height", height)]:
        if val is not None:
            s.CellsU(cell).FormulaU = f"{float(val)} in"
    return f"updated shape {shape_id}"


@mcp.tool()
def set_text(shape_id: int, text: str, font_size_pt: float | None = None, bold: bool | None = None) -> str:
    s = _shape(shape_id)
    s.Text = text
    if font_size_pt is not None:
        s.CellsU("Char.Size").FormulaU = f"{font_size_pt} pt"
    if bold is not None:
        s.CellsU("Char.Style").FormulaU = "1" if bold else "0"
    return f"updated text {shape_id}"


@mcp.tool()
def connect(shape_a: int, shape_b: int, color_rgb: str = "RGB(10,88,199)", dashed: bool = False) -> int:
    if PAGE is None:
        raise RuntimeError("No active page")
    app = _app()
    a, b = _shape(shape_a), _shape(shape_b)
    c = PAGE.Drop(app.ConnectorToolDataObject, 0, 0)
    c.CellsU("BeginX").GlueTo(a.CellsU("PinX"))
    c.CellsU("EndX").GlueTo(b.CellsU("PinX"))
    c.CellsU("LineColor").FormulaU = color_rgb
    c.CellsU("LineWeight").FormulaU = "1.5 pt"
    c.CellsU("EndArrow").FormulaU = "13"
    if dashed:
        c.CellsU("LinePattern").FormulaU = "9"
    return int(c.ID)


@mcp.tool()
def save(path: str | None = None) -> str:
    if DOC is None:
        raise RuntimeError("No active document")
    if path:
        DOC.SaveAs(str(Path(path).resolve()))
    else:
        DOC.Save()
    return str(DOC.FullName)


if __name__ == "__main__":
    mcp.run()
