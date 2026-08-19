# Level-C v04 输入文件与断点运行顺序

## 不上传GitHub的大体积/原始数据
以下文件保留在本地实验环境，不直接提交仓库：
- `month_expanded_2024_GRU-RNN.npz`：2024-05训练、2024-06测试的GRU真实预测/实测结果；
- `levelC_dataset_v03.npz`：由真实健康残差背景构造的半物理故障窗口；
- 原始DCS月度数据。

仓库中保存**可复现脚本、故障签名、邻接构造规则、实验结果CSV和论文状态文件**。

## 推荐断点顺序

### A. 数据再生成
```bash
python experiments/levelC_v04/generate_levelC_dataset.py \
  --gru-npz /path/month_expanded_2024_GRU-RNN.npz \
  --signature-csv experiments/levelC_v04/fault_signature_active_v03.csv \
  --out-dir work/levelC_v04
```

### B. 构建A0本机有序传感器KG
```bash
python experiments/levelC_v04/build_ordered_sensor_graph.py \
  --dataset work/levelC_v04/levelC_dataset_v03_regenerated.npz \
  --out work/levelC_v04/kg_ordered_sensor_adjacency.csv
```

### C. 每次只跑一个seed
```bash
python experiments/levelC_v04/train_ordered_sensor_gcn.py \
  --dataset work/levelC_v04/levelC_dataset_v03_regenerated.npz \
  --adjacency work/levelC_v04/kg_ordered_sensor_adjacency.csv \
  --seed 123 \
  --out work/levelC_v04/kg_seed123.csv \
  --cm work/levelC_v04/kg_seed123_cm.csv
```

建议seed：`101, 123, 202, 303, 404`。每个seed独立运行、独立保存，禁止把5个seed放进一次长命令。

### D. 汇总
```bash
python experiments/levelC_v04/summarize_results.py \
  work/levelC_v04/kg_seed101.csv \
  work/levelC_v04/kg_seed123.csv \
  work/levelC_v04/kg_seed202.csv \
  work/levelC_v04/kg_seed303.csv \
  work/levelC_v04/kg_seed404.csv \
  --model KG-OrderedSensor \
  --out work/levelC_v04/kg_5seed_summary.csv
```

## 重要实验边界
- 当前半物理故障数据不是现场真实故障；
- A0 KG邻接不使用 `fault_signature_active_v03.csv` 定义任何图边；
- 当前 0.8958 Macro-F1 仅是 development checkpoint，不是最终论文精度；
- Mask预训练现阶段是负结果，相关CSV仅用于研发追踪；
- 下一阶段提高精度时不得改变test split或用test挑超参数。
