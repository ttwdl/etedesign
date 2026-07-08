# 云端 Git 部署说明

本项目已经整理好适合上传到云端 Git 的轻量材料。

## 当前状态

- 本地 git 已经存在。
- 当前没有配置远端仓库。
- 本机没有安装 GitHub CLI，所以现在不能直接自动创建 GitHub 仓库。
- 大体积训练输出、checkpoint、`.npy` 数据仍然被 `.gitignore` 忽略，不建议直接上传到普通 git。

## 推荐上传内容

已经整理到：

`cloud_share/`

里面包括：

- `all_experiments_comparison.csv`：所有主要实验对比表。
- `external_inference_25ch_150_batch.csv`：当前最佳模型在多个外部 CAVE 场景上的推理结果。
- `best_25ch_150_structure.csv`：当前最佳 25 通道结构参数。
- `best_25ch_150_eval_angles.csv`：当前最佳模型角度评估。
- `best_25ch_150_fabrication_mc.csv`：当前最佳模型制造误差 Monte Carlo 评估。
- `figures/`：少量用于汇报和给网页版 AI 查看用的关键图片。

项目根目录还有：

- `experiment_summary_20260708.md`：实验结论总结。
- `cloud_share/WEB_AI_PROMPT.md`：给网页版 AI 的交流 prompt。

## 如果你已经在 GitHub 创建了空仓库

假设你的 GitHub 仓库地址是：

`https://github.com/你的用户名/你的仓库名.git`

在项目目录运行：

```powershell
git remote add origin https://github.com/你的用户名/你的仓库名.git
git push -u origin master
```

如果远端已经存在 `origin`，则改用：

```powershell
git remote set-url origin https://github.com/你的用户名/你的仓库名.git
git push -u origin master
```

## 如果 GitHub 要求登录

正常情况会弹出浏览器或 Git Credential Manager 登录窗口。

如果没有弹窗，建议安装 GitHub CLI 后执行：

```powershell
gh auth login
```

然后再 push。

## 不建议上传的内容

普通 git 不建议上传这些：

- `checkpoints*/`
- `results*/` 里面的大量 `.npy`
- `data_cache*/`
- CAVE 原始数据
- TensorBoard `runs/`

原因是这些文件很大，普通 GitHub 仓库会变慢，也不方便网页版 AI 阅读。

如果以后确实要云端保存完整 checkpoint 和数据，应该单独用：

- Git LFS
- OneDrive / 百度网盘 / Google Drive
- Zenodo / Figshare
- Hugging Face Datasets

当前这次先上传“代码 + 结论性结果 + 小图”，更适合和网页版 AI 交流。

