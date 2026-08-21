using System;
using System.IO;
using OfficeIMO.Visio;
using OfficeIMO.Visio.Fluent;
using OfficeIMO.Drawing;
using Color = OfficeIMO.Drawing.OfficeColor;

internal static class Program {
    private const double PageW = 16.50;
    private const double PageH = 9.27;
    private const double Px = 100.0;

    private static readonly Color Black = Color.FromRgb(25, 25, 25);
    private static readonly Color Gray = Color.FromRgb(95, 103, 112);
    private static readonly Color LightGray = Color.FromRgb(238, 242, 246);
    private static readonly Color FrameGray = Color.FromRgb(132, 138, 145);
    private static readonly Color Orange = Color.FromRgb(234, 82, 13);
    private static readonly Color PaleOrange = Color.FromRgb(255, 235, 219);
    private static readonly Color Blue = Color.FromRgb(26, 105, 177);
    private static readonly Color Navy = Color.FromRgb(8, 63, 135);
    private static readonly Color PaleBlue = Color.FromRgb(219, 234, 248);
    private static readonly Color TurbineBlue = Color.FromRgb(198, 215, 230);
    private static readonly Color Green = Color.FromRgb(78, 133, 52);
    private static readonly Color PaleGreen = Color.FromRgb(232, 242, 217);
    private static readonly Color White = Color.FromRgb(255, 255, 255);

    public static void Main() {
        string outDir = Environment.GetEnvironmentVariable("OUTPUT_DIR") ?? "out";
        Directory.CreateDirectory(outDir);
        string vsdxPath = Path.Combine(outDir, "汽轮机系统设备级状态监测建模_可编辑Visio.vsdx");

        VisioDocument document = VisioDocument.Create(vsdxPath);
        document.AsFluent()
            .Info(info => info
                .Title("汽轮机系统设备级状态监测建模示意图")
                .Author("KG-GCN turbine research project")
                .Subject("Editable Visio reconstruction and logic-corrected turbine system diagram"))
            .Page("01_原图可编辑复刻", PageW, PageH, VisioMeasurementUnit.Inches, BuildReplicaPage)
            .Page("02_逻辑修正版_论文推荐", PageW, PageH, VisioMeasurementUnit.Inches, BuildCorrectedPage)
            .End();

        document.Save();

        // Headless vector preview of the first page for GitHub artifact inspection.
        try {
            document.SaveAsSvg(Path.Combine(outDir, "汽轮机系统设备级状态监测建模_预览.svg"),
                new VisioSvgSaveOptions { PixelsPerInch = 110, BackgroundColor = White });
        } catch (Exception ex) {
            Console.WriteLine($"SVG preview skipped: {ex.Message}");
        }

        Console.WriteLine($"Generated: {vsdxPath}");
    }

    private static void PreparePage(VisioFluentPage p) {
        p.Margins(0.12)
         .Layer("系统边界")
         .Layer("设备")
         .Layer("蒸汽")
         .Layer("水系统")
         .Layer("疏水")
         .Layer("建模单元")
         .Layer("文字标注")
         .Layer("机械连接");
    }

    // ---------- coordinate helpers: source image uses top-left pixel coordinates ----------
    private static double CX(double leftPx, double widthPx) => (leftPx + widthPx / 2.0) / Px;
    private static double CY(double topPx, double heightPx) => PageH - (topPx + heightPx / 2.0) / Px;
    private static double W(double px) => px / Px;
    private static double H(double px) => px / Px;

    private static string Id(string prefix, string name) => prefix + "_" + name.Replace("-", "_").Replace(" ", "_");

    private static void AddText(VisioFluentPage p, string id, double l, double t, double w, double h,
        string text, double font = 11, bool bold = false, Color? color = null,
        VisioTextHorizontalAlignment hAlign = VisioTextHorizontalAlignment.Center,
        VisioTextVerticalAlignment vAlign = VisioTextVerticalAlignment.Middle) {
        p.TextBox(id, CX(l, w), CY(t, h), W(w), H(h), text, s => s
            .Font("Microsoft YaHei")
            .FontSize(font)
            .Bold(bold)
            .TextColor(color ?? Black)
            .TextAlignment(hAlign, vAlign)
            .Layer("文字标注"));
    }

    private static void AddRect(VisioFluentPage p, string id, double l, double t, double w, double h,
        Color fill, Color stroke, double lineWeight = 0.012, int pattern = 1, string? text = null,
        string layer = "设备") {
        p.Rect(id, CX(l, w), CY(t, h), W(w), H(h), text)
         .Shape(id, s => s.Fill(fill).Stroke(stroke, lineWeight, pattern).Layer(layer)
            .Font("Microsoft YaHei").FontSize(10).TextColor(Black)
            .TextAlignment(VisioTextHorizontalAlignment.Center, VisioTextVerticalAlignment.Middle));
    }

    private static void AddCircle(VisioFluentPage p, string id, double l, double t, double d,
        Color fill, Color stroke, double lineWeight = 0.012, string? text = null, string layer = "设备") {
        p.Circle(id, CX(l, d), CY(t, d), W(d), text)
         .Shape(id, s => s.Fill(fill).Stroke(stroke, lineWeight).Layer(layer)
            .Font("Microsoft YaHei").FontSize(10).TextColor(Black)
            .TextAlignment(VisioTextHorizontalAlignment.Center, VisioTextVerticalAlignment.Middle));
    }

    private static void AddTrapezoid(VisioFluentPage p, string id, double l, double t, double w, double h,
        Color fill, Color stroke, string? text = null) {
        p.Trapezoid(id, CX(l, w), CY(t, h), W(w), H(h), text)
         .Shape(id, s => s.Fill(fill).Stroke(stroke, 0.014).Layer("设备")
            .Font("Microsoft YaHei").FontSize(9).TextColor(Black)
            .TextAlignment(VisioTextHorizontalAlignment.Center, VisioTextVerticalAlignment.Middle));
    }

    private static void AddFrame(VisioFluentPage p, string id, double l, double t, double w, double h,
        Color stroke, string title, double titleX, double titleY, double titleW) {
        AddRect(p, id, l, t, w, h, White, stroke, 0.010, 2, null, "系统边界");
        AddText(p, id + "_title", titleX, titleY, titleW, 34, title, 15, true, stroke == Green ? Navy : Navy);
    }

    private static void Anchor(VisioFluentPage p, string id, double xPx, double yPx) {
        AddRect(p, id, xPx - 1, yPx - 1, 2, 2, White, White, 0.001, 0, null, "系统边界");
    }

    private static void Connect(VisioFluentPage p, string from, string to, Color color,
        VisioSide fromSide = VisioSide.Right, VisioSide toSide = VisioSide.Left,
        bool arrow = true, bool rightAngle = false, int pattern = 1, double weight = 0.018,
        string layer = "蒸汽") {
        p.Connect(from, to, fromSide, toSide, c => {
            if (rightAngle) c.RightAngle(); else c.Straight();
            c.LineColor(color).LineWeight(weight).LinePattern(pattern).Layer(layer);
            if (arrow) c.ArrowEnd(EndArrow.Triangle);
        });
    }

    private static void AddTurbine(VisioFluentPage p, string prefix, string key, double l, double t,
        string line1, string line2, bool addModel = true) {
        string shapeId = Id(prefix, key);
        AddTrapezoid(p, shapeId, l, t, 62, 58, TurbineBlue, Black);
        AddText(p, shapeId + "_label", l - 18, t - 58, 98, 48, line1 + "\n" + line2, 11, true);
        if (addModel) AddModelUnit(p, prefix, key, l - 39, 247, 140, 76, shapeId);
    }

    private static void AddReheater(VisioFluentPage p, string prefix, string key, double l, double t,
        string line1, string line2, bool addModel = true) {
        string shapeId = Id(prefix, key);
        AddCircle(p, shapeId, l, t, 50, PaleOrange, Orange, 0.016, "∿", "设备");
        p.Shape(shapeId, s => s.FontSize(19).Bold(false));
        AddText(p, shapeId + "_label", l - 25, t - 58, 100, 48, line1 + "\n" + line2, 11, true);
        if (addModel) AddModelUnit(p, prefix, key, l - 40, 247, 140, 76, shapeId);
    }

    private static void AddModelUnit(VisioFluentPage p, string prefix, string key,
        double l, double t, double w, double h, string equipmentId) {
        string root = Id(prefix, "model_" + key);
        AddRect(p, root + "_frame", l, t, w, h, White, FrameGray, 0.010, 2, null, "建模单元");
        AddRect(p, root + "_u", l + 5, t + 37, 23, 29, PaleGreen, Green, 0.010, 1, "U", "建模单元");
        AddRect(p, root + "_m", l + 45, t + 34, 50, 32, LightGray, Gray, 0.010, 1, "模型", "建模单元");
        AddRect(p, root + "_y", l + w - 30, t + 37, 23, 29, PaleOrange, Orange, 0.010, 1, "Y", "建模单元");
        AddRect(p, root + "_b", l + w/2 - 11, t + 7, 22, 23, White, Blue, 0.010, 1, "B", "建模单元");
        p.Shape(root + "_u", s => s.Bold(true).FontSize(9));
        p.Shape(root + "_m", s => s.Bold(true).FontSize(9));
        p.Shape(root + "_y", s => s.Bold(true).FontSize(9));
        p.Shape(root + "_b", s => s.Bold(true).FontSize(9));
        Connect(p, root + "_u", root + "_m", Black, VisioSide.Right, VisioSide.Left, true, false, 1, 0.010, "建模单元");
        Connect(p, root + "_m", root + "_y", Black, VisioSide.Right, VisioSide.Left, true, false, 1, 0.010, "建模单元");
        Connect(p, root + "_b", root + "_m", Black, VisioSide.Bottom, VisioSide.Top, true, false, 1, 0.010, "建模单元");
        Connect(p, equipmentId, root + "_b", Black, VisioSide.Bottom, VisioSide.Top, true, false, 2, 0.010, "建模单元");
    }

    private static void AddHeater(VisioFluentPage p, string id, double l, double t) {
        AddCircle(p, id, l, t, 38, PaleGreen, Green, 0.012, "∿", "设备");
        p.Shape(id, s => s.FontSize(15));
    }

    private static void AddPump(VisioFluentPage p, string id, double l, double t, string? label = null) {
        AddCircle(p, id, l, t, 42, PaleBlue, Navy, 0.014, "▶", "设备");
        p.Shape(id, s => s.FontSize(12));
        if (!string.IsNullOrWhiteSpace(label)) AddText(p, id + "_label", l - 35, t + 45, 112, 28, label!, 10.5, true);
    }

    private static void AddBoiler(VisioFluentPage p, string prefix, double l, double t) {
        string body = Id(prefix, "boiler");
        AddRect(p, body, l, t + 18, 96, 116, Color.FromRgb(178, 190, 202), Black, 0.015, 1, null, "设备");
        p.Ellipse(body + "_dome", CX(l, 96), CY(t, 96), W(96), H(54), null)
         .Shape(body + "_dome", s => s.Fill(Color.FromRgb(197, 207, 216)).Stroke(Black, 0.015).Layer("设备"));
        AddRect(p, body + "_furnace", l + 28, t + 45, 40, 55, Color.FromRgb(246, 141, 58), Black, 0.012, 1, "🔥", "设备");
        p.Shape(body + "_furnace", s => s.FontSize(17));
        AddText(p, body + "_label", l - 6, t + 137, 108, 28, "锅炉", 12, true);
    }

    private static void AddCondenser(VisioFluentPage p, string prefix, string key, double l, double t,
        string label, bool modelUnit = false) {
        string id = Id(prefix, key);
        AddRect(p, id, l, t + 22, 82, 76, Color.FromRgb(116, 167, 210), Navy, 0.015, 1, null, "设备");
        p.Ellipse(id + "_top", CX(l, 82), CY(t, 82), W(82), H(48), null)
         .Shape(id + "_top", s => s.Fill(Color.FromRgb(137, 181, 218)).Stroke(Navy, 0.015).Layer("设备"));
        AddText(p, id + "_coil", l + 17, t + 41, 48, 30, "≋≋", 14, false, Navy);
        AddText(p, id + "_label", l + 86, t + 75, 86, 32, label, 11, true, Black, VisioTextHorizontalAlignment.Left);
        if (modelUnit) AddModelUnit(p, prefix, key, l - 25, t + 108, 132, 72, id);
    }

    // =====================================================================================
    // PAGE 1 — editable reconstruction of the current raster image
    // =====================================================================================
    private static void BuildReplicaPage(VisioFluentPage p) {
        const string q = "R";
        PreparePage(p);
        AddText(p, q + "_title", 420, 6, 820, 50, "汽轮机系统设备级状态监测建模示意图", 20, true);

        AddFrame(p, q + "_topFrame", 195, 61, 1217, 279, FrameGray, "汽轮机本体", 690, 65, 260);
        AddFrame(p, q + "_regenFrame", 104, 397, 1165, 175, Green, "回热系统", 666, 397, 240);
        AddFrame(p, q + "_coldFrame", 1281, 397, 355, 269, Blue, "冷端系统", 1390, 398, 210);
        AddFrame(p, q + "_bfFrame", 236, 581, 750, 84, Blue, "给水/小机系统", 648, 628, 220);

        AddBoiler(p, q, 49, 175);
        AddCondenser(p, q, "condenser", 1468, 214, "凝汽器");

        AddTurbine(p, q, "VHP", 278, 158, "VHP", "超高压缸");
        AddReheater(p, q, "RH1", 458, 158, "RH1", "一次再热器");
        AddTurbine(p, q, "HP", 628, 158, "HP", "高压缸");
        AddReheater(p, q, "RH2", 800, 158, "RH2", "二次再热器");
        AddTurbine(p, q, "IP", 964, 158, "IP", "中压缸");
        AddTurbine(p, q, "LP_A", 1127, 158, "LP-A", "低压缸A");
        AddTurbine(p, q, "LP_B", 1290, 158, "LP-B", "低压缸B");

        // Main steam / reheat chain, matching the current image.
        Connect(p, Id(q,"boiler"), Id(q,"VHP"), Orange, VisioSide.Right, VisioSide.Left, true, true);
        Connect(p, Id(q,"VHP"), Id(q,"RH1"), Orange);
        Connect(p, Id(q,"RH1"), Id(q,"HP"), Orange);
        Connect(p, Id(q,"HP"), Id(q,"RH2"), Orange);
        Connect(p, Id(q,"RH2"), Id(q,"IP"), Orange);
        Connect(p, Id(q,"IP"), Id(q,"LP_A"), Orange);
        Connect(p, Id(q,"LP_A"), Id(q,"LP_B"), Orange);
        Connect(p, Id(q,"LP_B"), Id(q,"condenser"), Orange, VisioSide.Right, VisioSide.Top, true, true);

        // Orange extraction-steam bus in the current picture.
        double[] busX = { 303, 486, 660, 830, 997, 1157, 1318 };
        for (int i = 0; i < busX.Length; i++) Anchor(p, $"{q}_bus{i}", busX[i], 373);
        for (int i = 0; i < busX.Length - 1; i++) Connect(p, $"{q}_bus{i}", $"{q}_bus{i+1}", Orange, VisioSide.Right, VisioSide.Left, false, false);
        string[] eq = { Id(q,"VHP"), Id(q,"RH1"), Id(q,"HP"), Id(q,"RH2"), Id(q,"IP"), Id(q,"LP_A"), Id(q,"LP_B") };
        for (int i = 0; i < eq.Length; i++) Connect(p, eq[i], $"{q}_bus{i}", Orange, VisioSide.Bottom, VisioSide.Top, true, false);

        // High-pressure heaters.
        AddText(p, q+"_hphTitle", 309, 409, 250, 34, "高压加热器组(高加)", 11, true);
        AddRect(p, q+"_hphBox", 221, 439, 323, 80, White, Green, 0.009, 2, null, "系统边界");
        AddHeater(p, q+"_hph1", 234, 455);
        AddHeater(p, q+"_hph2", 333, 455);
        AddText(p, q+"_hphDots", 396, 458, 70, 30, "···", 15, true);
        AddHeater(p, q+"_hph3", 493, 455);

        // Deaerator.
        AddRect(p, q+"_deaBase", 618, 472, 70, 36, Color.FromRgb(139,169,202), Navy, 0.012, 1, null, "设备");
        AddRect(p, q+"_deaTop", 638, 437, 30, 36, Color.FromRgb(213,220,228), Black, 0.010, 1, "", "设备");
        AddText(p, q+"_deaLabel", 602, 511, 104, 26, "除氧器", 11, true);

        // Low-pressure heaters.
        AddText(p, q+"_lphTitle", 919, 410, 265, 34, "低压加热器组(低加)", 11, true);
        AddRect(p, q+"_lphBox", 843, 439, 330, 80, White, Green, 0.009, 2, null, "系统边界");
        AddHeater(p, q+"_lph1", 860, 455);
        AddHeater(p, q+"_lph2", 961, 455);
        AddText(p, q+"_lphDots", 1024, 458, 70, 30, "···", 15, true);
        AddHeater(p, q+"_lph3", 1128, 455);

        // Extraction drops from the common bus, placed to visually match the raster.
        Connect(p, q+"_bus0", q+"_hph1", Orange, VisioSide.Bottom, VisioSide.Top, true, true);
        Connect(p, q+"_bus1", q+"_hph2", Orange, VisioSide.Bottom, VisioSide.Top, true, true);
        Connect(p, q+"_bus2", q+"_deaTop", Orange, VisioSide.Bottom, VisioSide.Top, true, true);
        Connect(p, q+"_bus3", q+"_lph1", Orange, VisioSide.Bottom, VisioSide.Top, true, true);
        Connect(p, q+"_bus4", q+"_lph2", Orange, VisioSide.Bottom, VisioSide.Top, true, true);
        Connect(p, q+"_bus5", q+"_lph3", Orange, VisioSide.Bottom, VisioSide.Top, true, true);

        // Feedwater / condensate line as it appears in the current image.
        AddPump(p, q+"_feedPump", 524, 596, "给水泵");
        AddTrapezoid(p, q+"_bfpt", 326, 592, 54, 41, TurbineBlue, Black);
        AddText(p, q+"_bfptLabel", 279, 637, 150, 25, "给水泵汽轮机", 10.5, true);
        AddPump(p, q+"_condPump", 1127, 596, "凝结水泵");

        // Blue feedwater path (right -> left -> boiler).
        Connect(p, q+"_condPump", q+"_lph3", Blue, VisioSide.Top, VisioSide.Bottom, true, true, 1, 0.016, "水系统");
        Connect(p, q+"_lph3", q+"_lph2", Blue, VisioSide.Left, VisioSide.Right, true, false, 1, 0.016, "水系统");
        Connect(p, q+"_lph2", q+"_lph1", Blue, VisioSide.Left, VisioSide.Right, true, false, 1, 0.016, "水系统");
        Connect(p, q+"_lph1", q+"_deaBase", Blue, VisioSide.Left, VisioSide.Right, true, false, 1, 0.016, "水系统");
        Connect(p, q+"_deaBase", q+"_feedPump", Blue, VisioSide.Bottom, VisioSide.Top, true, true, 1, 0.016, "水系统");
        Connect(p, q+"_feedPump", q+"_hph3", Blue, VisioSide.Left, VisioSide.Bottom, true, true, 1, 0.016, "水系统");
        Connect(p, q+"_hph3", q+"_hph2", Blue, VisioSide.Left, VisioSide.Right, true, false, 1, 0.016, "水系统");
        Connect(p, q+"_hph2", q+"_hph1", Blue, VisioSide.Left, VisioSide.Right, true, false, 1, 0.016, "水系统");
        Connect(p, q+"_hph1", Id(q,"boiler"), Blue, VisioSide.Left, VisioSide.Bottom, true, true, 1, 0.016, "水系统");
        AddText(p, q+"_feedLabel", 16, 446, 78, 80, "锅炉\n给水", 10.5, true, Blue);

        // Green drain/condensate dashed line.
        Anchor(p, q+"_drainL", 257, 538);
        Anchor(p, q+"_drainR", 1165, 538);
        Connect(p, q+"_drainL", q+"_drainR", Green, VisioSide.Right, VisioSide.Left, true, false, 2, 0.014, "疏水");
        Connect(p, q+"_hph1", q+"_drainL", Green, VisioSide.Bottom, VisioSide.Top, false, true, 2, 0.012, "疏水");
        Connect(p, q+"_hph2", q+"_drainL", Green, VisioSide.Bottom, VisioSide.Top, false, true, 2, 0.012, "疏水");
        Connect(p, q+"_hph3", q+"_drainL", Green, VisioSide.Bottom, VisioSide.Top, false, true, 2, 0.012, "疏水");
        Connect(p, q+"_lph1", q+"_drainR", Green, VisioSide.Bottom, VisioSide.Top, false, true, 2, 0.012, "疏水");
        Connect(p, q+"_lph2", q+"_drainR", Green, VisioSide.Bottom, VisioSide.Top, false, true, 2, 0.012, "疏水");
        Connect(p, q+"_lph3", q+"_drainR", Green, VisioSide.Bottom, VisioSide.Top, false, true, 2, 0.012, "疏水");
        AddText(p, q+"_drainLabel", 945, 533, 170, 28, "疏水/凝结水", 10.5, true, Green);

        // Cold-end block as in the raster image (kept unchanged on this replica page).
        AddPump(p, q+"_circPump", 1339, 454, "循环水泵");
        AddTrapezoid(p, q+"_tower", 1520, 470, 86, 100, LightGray, Navy);
        AddText(p, q+"_towerLabel", 1502, 577, 130, 28, "冷却塔", 11, true);
        Connect(p, Id(q,"condenser"), q+"_circPump", Blue, VisioSide.Bottom, VisioSide.Top, true, true, 1, 0.016, "水系统");
        Connect(p, q+"_circPump", q+"_tower", Blue, VisioSide.Right, VisioSide.Left, true, true, 1, 0.016, "水系统");
        Connect(p, q+"_tower", q+"_circPump", Blue, VisioSide.Bottom, VisioSide.Bottom, true, true, 1, 0.016, "水系统");

        // Bottom legend / explanatory area.
        AddRect(p, q+"_legendFrame", 15, 686, 1621, 231, White, FrameGray, 0.010, 2, null, "系统边界");
        Anchor(p, q+"_split1a", 510, 692); Anchor(p, q+"_split1b", 510, 909);
        Anchor(p, q+"_split2a", 1005, 692); Anchor(p, q+"_split2b", 1005, 909);
        Connect(p, q+"_split1a", q+"_split1b", FrameGray, VisioSide.Bottom, VisioSide.Top, false, false, 2, 0.008, "系统边界");
        Connect(p, q+"_split2a", q+"_split2b", FrameGray, VisioSide.Bottom, VisioSide.Top, false, false, 2, 0.008, "系统边界");
        AddText(p, q+"_legTitle", 34, 697, 120, 28, "图例说明", 11.5, true, Black, VisioTextHorizontalAlignment.Left);
        AddText(p, q+"_legPipeTitle", 70, 729, 120, 24, "管线含义", 10.5, true);
        Anchor(p, q+"_legO1", 42, 780); Anchor(p, q+"_legO2", 90, 780);
        Connect(p, q+"_legO1", q+"_legO2", Orange, arrow:false, weight:0.016);
        AddText(p, q+"_legOt", 95, 763, 160, 34, "蒸汽/再热蒸汽", 10, false, Black, VisioTextHorizontalAlignment.Left);
        Anchor(p, q+"_legB1", 42, 815); Anchor(p, q+"_legB2", 90, 815);
        Connect(p, q+"_legB1", q+"_legB2", Blue, arrow:false, weight:0.016, layer:"水系统");
        AddText(p, q+"_legBt", 95, 798, 190, 34, "给水/凝结水/循环水", 10, false, Black, VisioTextHorizontalAlignment.Left);
        Anchor(p, q+"_legG1", 42, 850); Anchor(p, q+"_legG2", 90, 850);
        Connect(p, q+"_legG1", q+"_legG2", Green, arrow:false, pattern:2, weight:0.014, layer:"疏水");
        AddText(p, q+"_legGt", 95, 833, 160, 34, "疏水/凝结水", 10, false, Black, VisioTextHorizontalAlignment.Left);

        AddText(p, q+"_legendModelTitle", 573, 699, 360, 26, "设备级状态监测模型（U/B/Y）", 11.5, true);
        AddRect(p, q+"_legendU", 529, 780, 130, 55, PaleGreen, Green, 0.010, 1, "U  操作变量\n（调节量/运行量）", "建模单元");
        AddRect(p, q+"_legendM", 684, 780, 125, 55, LightGray, Gray, 0.010, 1, "模型", "建模单元");
        AddRect(p, q+"_legendY", 835, 780, 125, 55, PaleOrange, Orange, 0.010, 1, "Y  状态输出\n（温度、压力等）", "建模单元");
        AddRect(p, q+"_legendB", 665, 732, 160, 42, White, Blue, 0.010, 1, "B  边界条件\n（上游/外部边界）", "建模单元");
        Connect(p, q+"_legendU", q+"_legendM", Black, weight:0.010, layer:"建模单元");
        Connect(p, q+"_legendM", q+"_legendY", Black, weight:0.010, layer:"建模单元");
        Connect(p, q+"_legendB", q+"_legendM", Black, VisioSide.Bottom, VisioSide.Top, true, false, 1, 0.010, "建模单元");
        AddText(p, q+"_interface", 578, 855, 360, 32, "相邻设备接口：上一级 Y → 下一级 B", 10.5, true, Blue);

        AddText(p, q+"_notesTitle", 1025, 699, 140, 28, "说明", 11.5, true, Black, VisioTextHorizontalAlignment.Left);
        AddText(p, q+"_notes", 1032, 737, 565, 160,
            "• 主蒸汽自锅炉进入，依次通过各级缸膨胀并经再热器加热；\n"+
            "• 各级抽汽进入高/低压加热器及除氧器；\n"+
            "• 给水经凝结水泵、低加、除氧器、给水泵和高加后返回锅炉；\n"+
            "• 本页严格复刻当前图片的结构关系，逻辑修正见第2页。",
            9.5, false, Black, VisioTextHorizontalAlignment.Left, VisioTextVerticalAlignment.Top);
    }

    // =====================================================================================
    // PAGE 2 — physically corrected version recommended for the paper
    // =====================================================================================
    private static void BuildCorrectedPage(VisioFluentPage p) {
        const string q = "C";
        PreparePage(p);
        AddText(p, q + "_title", 380, 6, 900, 50, "汽轮机系统设备级状态监测建模示意图（逻辑修正版）", 19, true);

        AddFrame(p, q+"_mainFrame", 175, 62, 1205, 310, FrameGray, "主通流与再热系统", 650, 67, 310);
        AddFrame(p, q+"_regenFrame", 120, 405, 1050, 185, Green, "回热系统", 555, 408, 220);
        AddFrame(p, q+"_coldFrame", 1190, 390, 430, 290, Blue, "冷端系统", 1305, 395, 210);
        AddFrame(p, q+"_bfFrame", 250, 598, 735, 78, Blue, "给水/小机系统", 610, 628, 250);

        AddBoiler(p, q, 45, 180);
        AddText(p, q+"_boilerBoundary", 18, 127, 150, 48, "锅炉侧边界\n（主蒸汽/给水）", 10.5, true, Gray);

        AddTurbine(p, q, "VHP", 250, 165, "VHP", "超高压缸");
        AddReheater(p, q, "RH1", 425, 165, "RH1", "一次再热器");
        AddTurbine(p, q, "HP", 595, 165, "HP", "高压缸");
        AddReheater(p, q, "RH2", 770, 165, "RH2", "二次再热器");
        AddTurbine(p, q, "IP", 940, 165, "IP", "中压缸");

        // LP-A and LP-B are parallel branches, not series.
        AddTurbine(p, q, "LP_A", 1120, 115, "LP-A", "低压缸A", false);
        AddTurbine(p, q, "LP_B", 1120, 235, "LP-B", "低压缸B", false);
        AddModelUnit(p, q, "LP_A", 1067, 175, 125, 68, Id(q,"LP_A"));
        AddModelUnit(p, q, "LP_B", 1200, 175, 125, 68, Id(q,"LP_B"));

        Connect(p, Id(q,"boiler"), Id(q,"VHP"), Orange, VisioSide.Right, VisioSide.Left, true, true);
        Connect(p, Id(q,"VHP"), Id(q,"RH1"), Orange);
        Connect(p, Id(q,"RH1"), Id(q,"HP"), Orange);
        Connect(p, Id(q,"HP"), Id(q,"RH2"), Orange);
        Connect(p, Id(q,"RH2"), Id(q,"IP"), Orange);
        Connect(p, Id(q,"IP"), Id(q,"LP_A"), Orange, VisioSide.Right, VisioSide.Left, true, true);
        Connect(p, Id(q,"IP"), Id(q,"LP_B"), Orange, VisioSide.Right, VisioSide.Left, true, true);

        // Two condensers corresponding to the two LP exhaust paths.
        AddCondenser(p, q, "CND1", 1325, 100, "CND1\n凝汽器1", false);
        AddCondenser(p, q, "CND2", 1325, 225, "CND2\n凝汽器2", false);
        Connect(p, Id(q,"LP_A"), Id(q,"CND1"), Orange, VisioSide.Right, VisioSide.Left, true, false);
        Connect(p, Id(q,"LP_B"), Id(q,"CND2"), Orange, VisioSide.Right, VisioSide.Left, true, false);
        AddText(p, q+"_backpressure", 1430, 162, 180, 70, "背压作为\n低压缸/冷端边界 B", 10, true, Orange);

        // Regenerative heaters: distinct extraction groups rather than one fictitious common bus.
        AddText(p, q+"_hphTitle", 250, 418, 250, 32, "高压加热器组（高加）", 11, true);
        AddRect(p, q+"_hphBox", 220, 450, 330, 82, White, Green, 0.009, 2, null, "系统边界");
        AddHeater(p, q+"_hph1", 240, 469); AddHeater(p, q+"_hph2", 350, 469); AddHeater(p, q+"_hph3", 470, 469);
        AddText(p, q+"_deaTitle", 605, 418, 160, 32, "除氧器", 11, true);
        AddRect(p, q+"_deaBase", 635, 468, 82, 42, Color.FromRgb(139,169,202), Navy, 0.012, 1, null, "设备");
        AddRect(p, q+"_deaTop", 657, 443, 38, 28, Color.FromRgb(213,220,228), Black, 0.010, 1, null, "设备");
        AddText(p, q+"_lphTitle", 805, 418, 260, 32, "低压加热器组（低加）", 11, true);
        AddRect(p, q+"_lphBox", 790, 450, 335, 82, White, Green, 0.009, 2, null, "系统边界");
        AddHeater(p, q+"_lph1", 815, 469); AddHeater(p, q+"_lph2", 925, 469); AddHeater(p, q+"_lph3", 1045, 469);

        // Physically grouped extraction steam.
        Connect(p, Id(q,"VHP"), q+"_hph1", Orange, VisioSide.Bottom, VisioSide.Top, true, true);
        Connect(p, Id(q,"HP"), q+"_hph2", Orange, VisioSide.Bottom, VisioSide.Top, true, true);
        Connect(p, Id(q,"IP"), q+"_deaTop", Orange, VisioSide.Bottom, VisioSide.Top, true, true);
        Connect(p, Id(q,"LP_A"), q+"_lph1", Orange, VisioSide.Bottom, VisioSide.Top, true, true);
        Connect(p, Id(q,"LP_B"), q+"_lph3", Orange, VisioSide.Bottom, VisioSide.Top, true, true);

        // Condensate/feedwater: condenser -> condensate pump -> low heaters -> deaerator -> feedwater pump -> high heaters -> boiler.
        AddPump(p, q+"_condPump", 1198, 530, "凝结水泵");
        AddPump(p, q+"_feedPump", 525, 604, "给水泵");
        Connect(p, Id(q,"CND1"), q+"_condPump", Blue, VisioSide.Bottom, VisioSide.Top, true, true, 1, 0.016, "水系统");
        Connect(p, Id(q,"CND2"), q+"_condPump", Blue, VisioSide.Bottom, VisioSide.Right, true, true, 1, 0.016, "水系统");
        Connect(p, q+"_condPump", q+"_lph3", Blue, VisioSide.Left, VisioSide.Right, true, true, 1, 0.016, "水系统");
        Connect(p, q+"_lph3", q+"_lph2", Blue, VisioSide.Left, VisioSide.Right, true, false, 1, 0.016, "水系统");
        Connect(p, q+"_lph2", q+"_lph1", Blue, VisioSide.Left, VisioSide.Right, true, false, 1, 0.016, "水系统");
        Connect(p, q+"_lph1", q+"_deaBase", Blue, VisioSide.Left, VisioSide.Right, true, false, 1, 0.016, "水系统");
        Connect(p, q+"_deaBase", q+"_feedPump", Blue, VisioSide.Bottom, VisioSide.Top, true, true, 1, 0.016, "水系统");
        Connect(p, q+"_feedPump", q+"_hph3", Blue, VisioSide.Left, VisioSide.Bottom, true, true, 1, 0.016, "水系统");
        Connect(p, q+"_hph3", q+"_hph2", Blue, VisioSide.Left, VisioSide.Right, true, false, 1, 0.016, "水系统");
        Connect(p, q+"_hph2", q+"_hph1", Blue, VisioSide.Left, VisioSide.Right, true, false, 1, 0.016, "水系统");
        Connect(p, q+"_hph1", Id(q,"boiler"), Blue, VisioSide.Left, VisioSide.Bottom, true, true, 1, 0.016, "水系统");

        // Feedwater-pump turbine: mechanical drive, not a water-flow element.
        AddTrapezoid(p, q+"_bfpt", 350, 606, 58, 43, TurbineBlue, Black);
        AddText(p, q+"_bfptLabel", 295, 645, 170, 24, "给水泵汽轮机", 10.5, true);
        Anchor(p, q+"_shaftA", 408, 626); Anchor(p, q+"_shaftB", 525, 626);
        Connect(p, q+"_shaftA", q+"_shaftB", Black, arrow:false, weight:0.018, layer:"机械连接");
        Connect(p, Id(q,"IP"), q+"_bfpt", Orange, VisioSide.Bottom, VisioSide.Top, true, true, 1, 0.014, "蒸汽");
        Connect(p, q+"_bfpt", Id(q,"CND1"), Orange, VisioSide.Right, VisioSide.Bottom, true, true, 1, 0.014, "蒸汽");
        AddText(p, q+"_bfptNote", 270, 588, 220, 24, "驱动蒸汽 → 小机 → 凝汽器", 9.5, false, Gray);

        // Cooling-water circuit is independent from the condensate/feedwater circuit.
        AddPump(p, q+"_circPump", 1260, 500, "循环水泵");
        AddTrapezoid(p, q+"_tower", 1500, 505, 86, 112, LightGray, Navy);
        AddText(p, q+"_towerLabel", 1480, 622, 125, 25, "冷却塔", 11, true);
        Connect(p, q+"_tower", q+"_circPump", Blue, VisioSide.Left, VisioSide.Right, true, true, 1, 0.015, "水系统");
        Connect(p, q+"_circPump", Id(q,"CND1"), Blue, VisioSide.Top, VisioSide.Bottom, true, true, 1, 0.015, "水系统");
        Connect(p, q+"_circPump", Id(q,"CND2"), Blue, VisioSide.Top, VisioSide.Bottom, true, true, 1, 0.015, "水系统");
        Connect(p, Id(q,"CND1"), q+"_tower", Blue, VisioSide.Bottom, VisioSide.Top, true, true, 1, 0.015, "水系统");
        Connect(p, Id(q,"CND2"), q+"_tower", Blue, VisioSide.Bottom, VisioSide.Top, true, true, 1, 0.015, "水系统");

        // Drain line, deliberately kept separate from the main feedwater line.
        Anchor(p, q+"_drainL", 255, 555); Anchor(p, q+"_drainR", 1085, 555);
        Connect(p, q+"_drainR", q+"_drainL", Green, VisioSide.Left, VisioSide.Right, true, false, 2, 0.013, "疏水");
        Connect(p, q+"_hph1", q+"_drainL", Green, VisioSide.Bottom, VisioSide.Top, false, true, 2, 0.011, "疏水");
        Connect(p, q+"_lph3", q+"_drainR", Green, VisioSide.Bottom, VisioSide.Top, false, true, 2, 0.011, "疏水");
        AddText(p, q+"_drainLabel", 765, 542, 190, 26, "疏水/级联疏水", 9.5, true, Green);

        // Modeling interface legend and logic notes.
        AddRect(p, q+"_legendFrame", 20, 700, 1605, 207, White, FrameGray, 0.010, 2, null, "系统边界");
        AddText(p, q+"_legendTitle", 42, 709, 250, 26, "设备级建模关系", 11.5, true, Black, VisioTextHorizontalAlignment.Left);
        AddRect(p, q+"_legendU", 80, 770, 150, 58, PaleGreen, Green, 0.010, 1, "U  操作/调节变量\n（按设备可用测点）", "建模单元");
        AddRect(p, q+"_legendB", 275, 735, 190, 58, White, Blue, 0.010, 1, "B  边界条件\n（入口P/T/流量、背压等）", "建模单元");
        AddRect(p, q+"_legendM", 300, 810, 140, 52, LightGray, Gray, 0.010, 1, "状态监测模型", "建模单元");
        AddRect(p, q+"_legendY", 510, 770, 170, 58, PaleOrange, Orange, 0.010, 1, "Y  状态输出\n（设备状态测点）", "建模单元");
        Connect(p, q+"_legendU", q+"_legendM", Black, VisioSide.Right, VisioSide.Left, true, true, 1, 0.010, "建模单元");
        Connect(p, q+"_legendB", q+"_legendM", Black, VisioSide.Bottom, VisioSide.Top, true, false, 1, 0.010, "建模单元");
        Connect(p, q+"_legendM", q+"_legendY", Black, VisioSide.Right, VisioSide.Left, true, true, 1, 0.010, "建模单元");
        AddText(p, q+"_interfaceNote", 72, 858, 615, 32,
            "物理接口关系：上游设备 Y 与下游设备 B 对应；训练时仍使用真实 DCS 边界测量值。", 9.7, true, Blue);

        AddText(p, q+"_logicTitle", 755, 710, 180, 26, "本页逻辑修正", 11.5, true, Black, VisioTextHorizontalAlignment.Left);
        AddText(p, q+"_logicNotes", 755, 742, 820, 155,
            "① LP-A、LP-B 改为并联排汽支路，不再串联；\n"+
            "② 按项目对象拆分 CND1/CND2，并把背压作为低压缸/冷端边界；\n"+
            "③ 凝结水回热回路与循环冷却水回路彻底分开；\n"+
            "④ 给水泵汽轮机按机械驱动关系绘制，不放在给水管线上；\n"+
            "⑤ 抽汽按高加/除氧器/低加分组连接，取消虚构的统一抽汽母管；\n"+
            "⑥ U/B/Y 表示测点语义映射，避免暗示训练时必须用上游模型预测值级联。",
            9.6, false, Black, VisioTextHorizontalAlignment.Left, VisioTextVerticalAlignment.Top);
    }
}
