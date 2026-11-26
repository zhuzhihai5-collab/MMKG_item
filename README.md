# MMKG Item

MMKG Item 是一个基于 PyTorch 的多模态知识图谱补全实验项目。给定头实体与关系，模型将图谱结构嵌入、图像特征和文本特征融合，对候选尾实体排序。反向关系则把头实体预测转成同一套打分流程。

## 模型与评估

- **模态编码**使用可训练的结构嵌入和池化后的图文特征。每个模态通过门控网络为多个专家分配权重。
- **门控融合**联合结构、图像和文本表示。训练使用四路 BCE 损失，覆盖三种单模态打分与融合打分。
- **链接预测**在计算排名时过滤已知正例，输出 MR、MRR 和 Hits@1/3/10/100。当前训练脚本的周期评估使用测试集，汇总里的验证集字段仍是初始值。

## 快速体验

需要 Python 3.10 或更新版本。下面使用 PowerShell，安装后即可在 CPU 上运行合成数据示例。

```powershell
git clone https://github.com/zhuzhihai5-collab/MMKG_item.git
cd MMKG_item
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe smoke_demo.py
```

示例会在临时目录生成 16 个实体及随机图文特征，运行一轮 CPU 训练和评分，结束后清理临时文件。输出可用来检查训练和评估链路。

## DB15K 训练

仓库提供模型和训练脚本。请自行准备适用于该项目的 DB15K 划分与预计算特征。原始 DB15K 可参考 [MMKB](https://github.com/mniepert/mmkb)，补全实验常用的划分与特征来源可参考 [MMRNS](https://github.com/quqxui/MMRNS)，并在使用前检查文件格式和 ID 对齐。

```text
datasets/DB15K/
├── entity2id.txt      # 每行：实体 整数ID
├── relation2id.txt    # 每行：关系 整数ID
├── train.txt          # 每行：头实体 关系 尾实体
├── valid.txt
├── test.txt
├── img_features.pth   # [实体数, 4096]，按实体 ID 排列
└── text_features.pth  # [实体数, 768]，按实体 ID 排列
```

保存特征时使用 PyTorch 张量。映射 ID 应从 0 连续编号，三元组按空格分隔。按你的显卡驱动和 CUDA 版本安装匹配的 PyTorch，然后从仓库根目录运行以下命令。

```powershell
.\.venv\Scripts\python.exe train.py --dataset DB15K --cuda 0 --epochs 100 --eval_freq 100 --batch_size 256
```

- `--cuda -1` 可使用 CPU。显存紧张时先减小 `--batch_size`。
- 保持默认的 `--dim 256 --r_dim 256` 和图文模态开启，以匹配当前模型的池化和打分维度。
- 默认会保存最后一轮权重到 `checkpoint/DB15K/trained_model.pth`。加 `--save 0` 可跳过保存，也可用 `--decoder_save_model` 指定文件路径。
- 正式实验应使用验证集选择超参和 checkpoint，并在参数固定后报告测试结果。

## 代码入口

| 文件 | 作用 |
| --- | --- |
| [`train.py`](train.py) | 参数解析、训练、周期评估、权重保存 |
| [`models/Multi_MoE.py`](models/Multi_MoE.py) | 三模态表示、融合、候选实体打分 |
| [`layers/`](layers) | MoE 专家网络和模态注意力融合 |
| [`utils/data_util.py`](utils/data_util.py) | ID 映射、三元组和预计算特征加载 |
| [`utils/data_loader.py`](utils/data_loader.py) | 批构造、反向关系和过滤排名 |
