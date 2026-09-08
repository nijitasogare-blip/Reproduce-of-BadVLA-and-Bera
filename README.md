# VLA Backdoor & Defense Reproduction Suite

复现论文：
- BadVLA: Towards Backdoor Attacks on Vision-Language-Action Models via Objective-Decoupled Optimization (NeurIPS 2025)
- Bera: When Attention Betrays: Erasing Backdoor Attacks in Robotic Policies by Reconstructing Visual Tokens (ICRA 2026)

## 1. 实验设备与环境
- 笔记本：RTX 5060 Laptop 8GB（训练/推理）
- Docker Desktop（WSL2）
- 容器内 Python 3.11.10, torch 2.13.0+cu130, transformers 4.40.1, bitsandbytes 0.49.2, peft 0.11.1, tensorflow 2.15.0, tensorflow-datasets 4.9.3, LIBERO (PyPI 0.1.1), robosuite 1.4.0, mujoco 2.3.7
- 关键环境变量：LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cu13/lib（bnb 需要）；HF_ENDPOINT=https://hf-mirror.com

## 2. 模型与数据集
- 基座：moojink/openvla-7b-oft-finetuned-libero-spatial（本地 /data/models/...）
- 训练数据：openvla/modified_libero_rlds 的 libero_spatial_no_noops（RLDS）
- 触发器：画面中心约 10% 白色块（BadVLA 像素触发器）
- 评测：官方 LIBERO-Spatial 协议（10 任务，每个任务官方初始状态，220 步上限）

## 3. 目录结构
- attack/：BadVLA Stage-I/II 训练（QLoRA 单卡适配版）与自动流水线（r3a/r3c）
- eval/ 与 rollout/：官方评测（attack_eval_run.py）、对比视频（compare_videos.py、record_bera_video.py）、回放
- defense/：Bera 防御模块（FBL、AFM、解码器、三组评测、步数扫描）
- monitor/：训练监视脚本
- docs/：实验报告与机制验证报告

## 4. 关键参数（可复现结果）
Stage-I：LoRA r=4（投影/特征层），lr=5e-4，image_aug=True，每 2 步存点，早停阈 -0.55，取 step14。
Stage-II：LoRA r=4（Llama q/k/v/o），lr=5e-5，6000 步，image_aug=True。
Bera 解码器：UNet(32/64/128)，lr=1e-3，1600 步（白块训练）。
评测：trials=3/任务（10 任务=30 集）。

## 5. 快速运行示例（容器内）
```bash
# 官方评测（干净） 用于验证模型基座是否安装正确
python /workspace/.../src/eval/attack_eval_run.py --checkpoint /data/models/openvla-7b-oft-finetuned-libero-spatial --adapter <final_backdoor_adapter> --suite libero_spatial --trials_per_task 3
# 触发器评测：命令加 --trigger 
# 进行后门训练
python src/attack/r3a_pipeline.py \
  --round-root /data/runs/attack/my_run \
  --checkpoint /data/models/openvla-7b-oft-finetuned-libero-spatial \
  --stage2-steps 6000
# Bera 三组评测
GROUPS=nodefense,bera,random DECODER=<dec.pt> python ./defense/bera_eval3.py
# 监视训练
python ./monitor/train_monitor.py --log <log>
```
（脚本内默认 /data 路径，可按实际挂载修改。）

## 6. 版本依赖与 Git 上传说明
- 核心依赖版本已在上文列出
- 权重与数据因体积不入库：README 只记录 HF 源与本地路径。
- 本仓库不包含任何后门模型权重，只包含训练/评测代码与结果记录。

## 7. 主要结果速览
- BadVLA：干净 90%，ASR 100%
- Bera 防御试点：无防御 0/3 -> Bera 1/3（优于随机掩码 0/3）
- 详细数据见 docs/EXPERIMENT_REPORT.md
