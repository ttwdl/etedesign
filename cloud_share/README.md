# cloud_share 说明

这个目录是专门给云端 Git 和网页版 AI 准备的轻量结果包。

它不是完整训练输出，也不包含大体积 `.npy`、checkpoint 或 CAVE 原始数据。

## 最重要结论

当前最佳实验是：

`25ch_t06_tor20_150`

理由：

- 正式 test MSE 最低：`1.53e-4`
- 正式 test SAM 最低：`0.0655`
- 外部 CAVE 场景推理整体最好
- 结构满足当前工艺约束，最大深宽比约 `6.68`，低于限制 `10`

## 文件说明

| 文件 | 含义 |
|---|---|
| `all_experiments_comparison.csv` | 所有主要实验的总对比 |
| `external_inference_25ch_150_batch.csv` | 最佳 25 通道模型在 8 个外部 CAVE 场景上的推理结果 |
| `best_25ch_150_structure.csv` | 最佳 25 通道结构参数 |
| `best_25ch_150_eval_angles.csv` | 最佳 25 通道角度评估 |
| `best_25ch_150_fabrication_mc.csv` | 最佳 25 通道制造误差评估 |
| `WEB_AI_PROMPT.md` | 复制给网页版 AI 的 prompt |
| `figures/` | 关键示意图、光谱图和外部推理图 |

## 关键图片

| 图片 | 用途 |
|---|---|
| `figures/best_25ch_design_schematic.png` | 当前最佳结构示意图 |
| `figures/best_25ch_eval_spectra_0deg.png` | 最佳模型的滤光片透过谱 |
| `figures/network_architecture_ppt.png` | 网络结构图 |
| `figures/external_beads_selected_spectra.png` | 外部场景中误差较大的例子 |
| `figures/external_cloth_selected_spectra.png` | 外部场景中表现较好的例子 |
| `figures/external_superballs_selected_spectra.png` | 外部场景中 MSE 最低的例子 |

## 后续优化建议

优先路线：

`25ch + T_target=0.6 + 更强通道差异约束 + 150 epoch`

优化重点不是单纯增加通道数，而是让不同滤光片的透过谱更不一样、更有调制度。

