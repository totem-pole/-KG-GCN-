import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const root = path.dirname(fileURLToPath(import.meta.url));
const v24 = path.join(root, "58_v24_fair_comparison");
const outputDir = path.join(v24, "output", "xlsx");
const previewDir = path.join(v24, "tmp", "xlsx_previews");
await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const workbook = Workbook.create();
const readme = workbook.worksheets.add("说明与协议");
readme.showGridLines = false;
readme.getRange("A1:H1").merge();
readme.getRange("A1").values = [["V24 超高压缸 27 输出算法与机理融合对比"]];
readme.getRange("A1:H1").format = {
  fill: "#0F4C5C",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  rowHeight: 30,
  verticalAlignment: "center",
};
readme.getRange("A3:B16").values = [
  ["项目", "内容"],
  ["输入", "16 个 VHP 因果边界，不含凝汽器背压，不含转速"],
  ["输出", "27 个有效点；两个结构性不可观测疏水温度不纳入主模型"],
  ["四算法", "ANN / GRU-RNN / TCN / iTransformer-small"],
  ["机理对比", "GRU-RNN / PINN-GRU；预测结构一致，物理只进入训练 Loss"],
  ["月度协议1", "2024-05 训练 -> 2024-06 测试"],
  ["月度协议2", "2025-08 训练 -> 2025-09 测试"],
  ["长时域协议1", "2024-05-01起训练10d -> 后续100/200/300d"],
  ["长时域协议2", "2025-08-01起训练10d -> 后续100/200d；数据不足以构成300d"],
  ["防泄漏", "不输入真实输出历史；Scaler只拟合训练段；时间顺序切分"],
  ["热时间尺度", "温变起点滞后0-10min，金属松弛为小时级；V24需要快慢双时标"],
  ["术语核查", "项目内无独立 GTU-RNN；GRU 与 GRU-RNN 是同一基线，机理组为 PINN-GRU"],
  ["指标", "逐点 R² / RMSE / MAE / 训练最小最大值 / 峰值偏差 / 上1%偏差"],
  ["生成日期", "2026-08-15"],
];
readme.getRange("A3:B3").format = { fill: "#2A6F7F", font: { bold: true, color: "#FFFFFF" } };
readme.getRange("A3:B16").format.borders = { preset: "inside", style: "thin", color: "#D9E2E8" };
readme.getRange("A4:A16").format = { fill: "#EAF3F5", font: { bold: true, color: "#173F4F" } };
readme.getRange("A18:B23").values = [
  ["一手文献", "URL"],
  ["3D汽轮机缸体瞬态温度场", "https://doi.org/10.1016/0020-7403(93)90003-D"],
  ["汽轮机启动热-结构分析", "https://doi.org/10.1115/1.4049502"],
  ["汽轮机瞬态全共轭换热", "https://doi.org/10.1016/j.ijthermalsci.2017.10.025"],
  ["蒸汽容积动态时间常数", "https://doi.org/10.1049/joe.2016.0178"],
  ["备注", "提交论文前应再与原文全文逐条核对。"],
];
readme.getRange("A18:B18").format = { fill: "#2A6F7F", font: { bold: true, color: "#FFFFFF" } };
readme.getRange("A18:B23").format.borders = { preset: "inside", style: "thin", color: "#D9E2E8" };
readme.getRange("A:B").format.wrapText = true;
readme.getRange("A:A").format.columnWidth = 24;
readme.getRange("B:B").format.columnWidth = 92;

const csvSheets = [
  ["四算法任务汇总", path.join(v24, "four_models", "same_panel_comparisons", "V24_四算法_任务级汇总.csv")],
  ["四算法逐点指标", path.join(v24, "four_models", "same_panel_comparisons", "V24_27测点_四算法_逐任务指标.csv")],
  ["GRU_PINN任务汇总", path.join(v24, "pinn_gru", "same_panel_comparisons", "V24_GRU对PINNGRU_任务级汇总.csv")],
  ["GRU_PINN逐点指标", path.join(v24, "pinn_gru", "same_panel_comparisons", "V24_27测点_GRU对PINNGRU_逐任务指标.csv")],
  ["GRU_PINN配对差值", path.join(v24, "pinn_gru", "same_panel_comparisons", "V24_GRU对PINNGRU_逐点配对差值.csv")],
  ["网络结构", path.join(v24, "four_models", "same_panel_comparisons", "V24_四算法_网络结构与参数.csv")],
  ["热时间尺度", path.join(v24, "thermal_timescale_audit", "thermal_timescale_summary.csv")],
];

function columnLetter(indexZeroBased) {
  let value = indexZeroBased + 1;
  let result = "";
  while (value > 0) {
    value -= 1;
    result = String.fromCharCode(65 + (value % 26)) + result;
    value = Math.floor(value / 26);
  }
  return result;
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  const source = text.replace(/^\uFEFF/, "");
  for (let i = 0; i < source.length; i += 1) {
    const ch = source[i];
    if (quoted) {
      if (ch === '"' && source[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (ch === '"') {
        quoted = false;
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ',') {
      row.push(field);
      field = "";
    } else if (ch === '\n') {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += ch;
    }
  }
  if (field.length || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  const width = Math.max(...rows.map((item) => item.length));
  return rows.map((item, rowIndex) => {
    const padded = [...item, ...Array(width - item.length).fill("")];
    if (rowIndex === 0) return padded;
    return padded.map((value) => {
      const trimmed = value.trim();
      if (trimmed === "" || /^(nan|na|null)$/i.test(trimmed)) return "";
      if (/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(trimmed)) return Number(trimmed);
      return value;
    });
  });
}

for (let sheetIndex = 0; sheetIndex < csvSheets.length; sheetIndex += 1) {
  const [sheetName, csvPath] = csvSheets[sheetIndex];
  const csvText = await fs.readFile(csvPath, "utf8");
  const csvValues = parseCsv(csvText);
  const sheet = workbook.worksheets.add(sheetName);
  sheet.getRangeByIndexes(0, 0, csvValues.length, csvValues[0].length).values = csvValues;
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  const used = sheet.getUsedRange(true);
  used.format.font = { size: 9, color: "#1F2937" };
  const header = used.getRow(0);
  header.format = {
    fill: "#0F4C5C",
    font: { bold: true, color: "#FFFFFF", size: 9 },
    rowHeight: 28,
    verticalAlignment: "center",
    wrapText: true,
  };
  used.format.borders = {
    insideHorizontal: { style: "thin", color: "#E5E7EB" },
    bottom: { style: "thin", color: "#CBD5E1" },
  };
  used.format.autofitColumns();
  const headerValues = header.values[0];
  const rowCount = used.rowCount;
  const colCount = used.columnCount;
  for (let col = 0; col < colCount; col += 1) {
    const label = String(headerValues[col] ?? "");
    const range = sheet.getRangeByIndexes(1, col, Math.max(1, rowCount - 1), 1);
    if (/qualified|samples|outputs|epoch|count|year|horizon|lag/i.test(label)) {
      range.format.numberFormat = "#,##0";
    } else if (/r2|macro_r2|median_r2|delta_r2/i.test(label)) {
      range.format.numberFormat = "0.0000";
      range.conditionalFormats.add("colorScale", {
        colors: ["#FECACA", "#FEF3C7", "#BBF7D0"],
        thresholds: ["min", { type: "num", value: 0.8 }, "max"],
      });
    } else if (/rmse|mae|bias|error|actual|min|max|coefficient|tau|corr|skill/i.test(label)) {
      range.format.numberFormat = "0.000";
    }
    const preferred = /architecture|dropped|targets|live_inputs/i.test(label) ? 50
      : /name_cn|pair_name|task|target_tag/i.test(label) ? 28
      : /time|protocol|algorithm|group/i.test(label) ? 20
      : 14;
    sheet.getRangeByIndexes(0, col, rowCount, 1).format.columnWidth = preferred;
  }
  used.format.wrapText = false;
  const tableRange = `A1:${columnLetter(colCount - 1)}${rowCount}`;
  const table = sheet.tables.add(tableRange, true, `V24Table${sheetIndex + 1}`);
  table.style = "TableStyleMedium2";
  table.showFilterButton = true;

  const previewRows = Math.min(rowCount, 30);
  const previewCols = Math.min(colCount, 14);
  const preview = await workbook.render({
    sheetName,
    range: `A1:${columnLetter(previewCols - 1)}${previewRows}`,
    scale: 1.2,
    format: "png",
  });
  await fs.writeFile(path.join(previewDir, `${String(sheetIndex + 1).padStart(2, "0")}_${sheetName}.png`), new Uint8Array(await preview.arrayBuffer()));
}

const readmePreview = await workbook.render({ sheetName: "说明与协议", range: "A1:H23", scale: 1.3, format: "png" });
await fs.writeFile(path.join(previewDir, "00_说明与协议.png"), new Uint8Array(await readmePreview.arrayBuffer()));

const inspect = await workbook.inspect({
  kind: "table",
  range: "四算法任务汇总!A1:H20",
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 10,
  maxChars: 6000,
});
console.log(inspect.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
const outputPath = path.join(outputDir, "V24_27输出_四算法与PINNGRU_逐点结果.xlsx");
await xlsx.save(outputPath);
console.log(outputPath);
