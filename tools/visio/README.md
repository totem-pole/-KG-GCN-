# KG-GCN 论文框架图：Visio 原生编辑工作流

这个目录用于把 `figures/ch3/fig3_3_kg_gcn_framework_editable.svg` 转换为 **Microsoft Visio 原生可编辑形状**，并提供一个小型 Visio MCP 服务，用于后续逐形状微调。

## 1. 一键生成 VSDX

仅支持 Windows（需要安装桌面版 Microsoft Visio）。

```powershell
pip install pywin32
python tools/visio/svg_to_visio.py figures/ch3/fig3_3_kg_gcn_framework_editable.svg figures/ch3/fig3_3_kg_gcn_framework_editable.vsdx
```

输出 `.vsdx` 中，矩形、圆、椭圆、文字、直线和折线均为 Visio 原生对象，可单独移动和修改。

## 2. 启动 Visio MCP

```powershell
pip install mcp pywin32
python tools/visio/visio_mcp_server.py
```

服务提供：`create_blank`、`open_vsdx`、`list_shapes`、`move_resize`、`set_text`、`connect`、`save`。

连接到支持 MCP 的本地 Agent 后，可按 shape ID 一点点移动、改尺寸、改文字和重新连接。

## 3. 当前目标图

目标图为“基于KG-GCN的汽轮机系统故障诊断总体框架”。SVG 保持 1672×941 的原始比例。建议以 SVG 为视觉基准，VSDX 作为最终可编辑源文件。

> GitHub 可以保存和迭代源文件，但真正的 Visio 窗口操作仍需要在装有 Visio 的 Windows 机器上启动 MCP 服务。