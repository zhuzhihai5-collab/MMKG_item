import torch
import torch.nn as nn
class ModalFusionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, multi, img_dim, txt_dim):
        super(ModalFusionLayer, self).__init__()

        self.in_dim = in_dim # 模态1维度
        self.out_dim = out_dim # 输出统一维度
        self.multi = multi   # 融合单元的组数
        self.img_dim = img_dim # 图像模态维度
        self.text_dim = txt_dim # 文本模态维度

        # self.multi是几个mlp叠加，与多模态无关，类似于多头注意力机制的作用
        modal1 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(in_dim, out_dim)
            modal1.append(nn.Sequential(do, lin, nn.ReLU()))  # 相当于简写了：输入 → Dropout → Linear → ReLU → 输出
        self.modal1_layers = nn.ModuleList(modal1)

        modal2 = []
        for _ in range(self.multi): #构建 multi 个 独立的 MLP
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(self.img_dim, out_dim)
            modal2.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal2_layers = nn.ModuleList(modal2)

        modal3 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(self.text_dim, out_dim)
            modal3.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal3_layers = nn.ModuleList(modal3)

        self.ent_attn = nn.Linear(self.out_dim, 1, bias=False)
        self.ent_attn.requires_grad_(True)

    def forward(self, modal1_emb, modal2_emb, modal3_emb):
        # emb:[b,dim]
        batch_size = modal1_emb.size(0)  #取 batch 大小。
        x_mm = []
        for i in range(self.multi):  # self.multi是几个mlp叠加，与多模态无关  经过multi个线性层  三个模态分别过第 i 个 MLP
            x_modal1 = self.modal1_layers[i](modal1_emb)  # [b,XX_dim]*[XX_dim,out_dim]=[b,out_dim]
            x_modal2 = self.modal2_layers[i](modal2_emb)  # [b,dim]
            x_modal3 = self.modal3_layers[i](modal3_emb)  # [b,dim]
            #先通过 Linear 把不同维度变成相同维度    再 stack 成 [b, 3, out_dim]    然后做注意力融合
            #每个模态经过这几个线性层之后维度就一致了 就可以堆叠了  之前文本模态[b,text_dim]  图像模态[b,img_dim]  结构模态[b,str_dim]
            x_stack = torch.stack((x_modal1, x_modal2, x_modal3), dim=1)  # 沿dim=1进行堆叠[b,3,dim]
            attention_scores = self.ent_attn(x_stack).squeeze(-1)  # [b,3] #计算注意力分数计算注意力分数 现在每个样本有 3 个分数。
            attention_weights = torch.softmax(attention_scores, dim=-1)  # [b,3]  #把 3 个分数变成概率：如某样本[0.6, 0.3, 0.1] 就说明模态一更重要
            context_vectors = torch.sum(attention_weights.unsqueeze(-1) * x_stack, dim=1)  # [b,dim]  扩维，加权求和
            x_mm.append(context_vectors) #存入列表
        x_mm = torch.stack(x_mm, dim=1)  # [b,multi,dim]
        x_mm = x_mm.sum(1).view(batch_size, self.out_dim)  # [b,dim]
        # x_mm = torch.relu(x_mm)
        return x_mm, attention_weights  # [b,dim]  [b,3]

