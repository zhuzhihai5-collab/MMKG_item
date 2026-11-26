import torch
import torch.nn as nn
import torch.nn.functional as F
#MoE 层由多个专家子网络和一个门控网络组成。门控网络根据输入生成专家权重，
# 通过 softmax 归一化后对多个专家输出进行加权求和，从而提升模型表达能力并增强特征解耦能力。
class PWLayer(nn.Module): #Dropout + 全连接层（Linear）
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.lin = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.lin(self.dropout(x)) #这一层本质上就是一个简单的 MLP 子网络。

class MoEAdaptorLayer(nn.Module): #不用一个神经网络，而是用多个“专家网络”，再通过一个 gate 网络决定每个专家的权重。
    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        super(MoEAdaptorLayer, self).__init__()
        self.n_exps = n_exps  #定义有几个专家
        self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)])  #三个专家对应三个神经网络
        self.gate=nn.Linear(layers[0],n_exps) #[dim,3]   #门控网络    输入维度 = dim  输出维度 = 专家数量    作用：计算每个专家的权重
    def forward(self, x):
        # x:[b,dim],  r=[b,1]
        gates = F.softmax(self.gate(x),dim=-1)  # x:[b,dim]*gate：[dim,3]=[b,3] 计算 gate 权重  比如[0.2, 0.5, 0.3]  说明第二个专家最重要 第三个次之
        # expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)]  # [(B, 1, D)]*n_exps 跟下面的for循环等价
        expert_outputs=[]
        for i in range(self.n_exps):  #依次遍历专家，然后每个专家都过这个x，最后放到这个列表里面
            tmp=self.experts[i](x) #[b,dim]
            tmp=tmp.unsqueeze(-2) #[b,1,dim]  #在倒数第二个位置扩展一维,扩展的维度默认是1
            expert_outputs.append(tmp)

        expert_outputs = torch.cat(expert_outputs, dim=-2)  # (B, 3, D)  # 在倒数第二个维度将三个专家的输出拼接，倒数第二个维度就是3了
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs  # [b,专家数量,1]*[b,专家数量,d]=[b,专家数量,d]  (小知识点：在向量矩阵相乘时1会变成d，叫做广播)  gets也扩展一个维度成了[b,3,1] Gate 加权
        return multiple_outputs.sum(dim=-2), expert_outputs, gates  # (b,D),(B, n_exps, D),(B, n_exps)


