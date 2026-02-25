# Brainseg 使用指南

本项目实现基于在线数据增强和置信度伪标签的 one-shot 脑 MRI 分割方法。

## 数据格式

原始数据目录结构：
```
data_root/
├── labeled/
│   ├── image/
│   │   ├── 000_atlas_43_bl.nii.gz  (atlas)
│   │   └── *.nii.gz
│   └── label/
│       ├── 000_atlas_43_bl.nii.gz
│       └── *.nii.gz
└── unlabeled/
    └── image/
        └── *.nii.gz  (用于训练，标签不参与训练)
```

- Atlas: `000_atlas_43_bl.nii.gz`
- 标签范围: 0 (背景), 1-138 (共139类)

## 完整运行流程

### Step 1: 预处理

预处理脚本会：
1. 遍历所有图像，找到非零区域边界
2. 裁剪全零切片
3. 填充到统一的尺寸（8的倍数）
4. 保存变换参数用于逆变换

```bash
python preprocess.py --data_root /path/to/your/data --output_root /path/to/processed/data
```

### Step 2: 训练

```bash
python train_nifti.py --datapath /path/to/processed/data
```

常用参数：
- `--gpu`: GPU编号 (默认: 0)
- `--n_iter`: 迭代次数 (默认: 10001)
- `--n_save_iter`: 模型保存频率 (默认: 1000)
- `--model_dir`: 模型输出目录 (默认: CANDI_Model)

训练过程中会在 `CANDI_Model/` 目录保存模型权重文件。

### Step 3: 测试

```bash
python test_nifti.py \
    --datapath /path/to/processed/data \
    --model_path CANDI_Model/10000.pth \
    --output_dir predictions
```

输出：
- `predictions/`: 预测结果 (与label文件夹格式相同)
- `res.npy`: Dice scores

## 网络要求

输入尺寸必须是 **8的倍数**。预处理脚本会自动处理此问题。

## 文件说明

| 文件 | 功能 |
|------|------|
| `preprocess.py` | 预处理：裁剪零切片 + 填充到统一尺寸 |
| `datagenerators_nifti.py` | 数据加载器 |
| `train_nifti.py` | 训练脚本 |
| `test_nifti.py` | 测试脚本（保存预测结果） |
| `network.py` | 网络架构 (3D U-Net + Registration) |
| `losses.py` | 损失函数 (NCC, Gradient, Entropy) |
