import torch
import torch.nn as nn

from layers.layer1_moe import MoEAdaptorLayer
from layers.layer2_fuse import ModalFusionLayer
from .model import BaseModel


# 结构 embedding ─┐
# 图像 embedding ──┼→ MoE → 模态融合 → 打分 → 所有实体排序
# 文本 embedding ─┘
#
# 本模型是一个基于多模态 MoE 机制的知识图谱补全模型。首先分别对结构、图像和文本模态进行 embedding，
# 并通过 Mixture-of-Experts 机制进行特征解耦和增强。随后采用门控融合策略整合三模态信息，
# 最终通过类似 DistMult 的双线性打分函数对所有实体进行评分，并使用 BCE 损失进行训练。
class Multi_MoE(BaseModel):
    def __init__(self, args):
        super(Multi_MoE, self).__init__(args)
        self.dim=args.dim
        self.device=args.device
        # 1、初始化结构embedding  这是标准的知识图谱 embedding 方法（类似 TransE 思路）。
        self.entity_embeddings = nn.Embedding(
            len(args.entity2id),
            args.dim,
            padding_idx=None
        )   #给每个实体一个向量
        nn.init.xavier_normal_(self.entity_embeddings.weight)

        self.relation_embeddings = nn.Embedding(
            2 * len(args.relation2id),  #构造了反向关系
            args.r_dim,
            padding_idx=None
        )#给每个关系一个向量
        nn.init.xavier_normal_(self.relation_embeddings.weight)

        # 3、对这些预处理的图文embedding进行维度统一处理
        if args.dataset == "DB15K":  #这里就用了一个数据集  对原始图像 / 文本特征做池化降维
            img_pool = torch.nn.AvgPool2d(4, stride=4)
            img = img_pool(args.img.to(self.device).view(-1, 64, 64))
            img = img.view(img.size(0), -1)
            txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            txt = txt_pool(args.desp.to(self.device).view(-1, 12, 64))
            txt = txt.view(txt.size(0), -1)


        #图像和文本 embedding 是预训练好的
        # 图像实体embedding使用图像本身的，图像关系embedding初始化一个
        # ps:区分CV训练的模型是为了得出图像embedding，而这里的图像embedding作为一个最底层存储embedding层，用于三元组对应匹配
        # 这几块儿一直都是拿到图像embedding，不涉及到图像embedding的CV模型

        #就是说拿别人已经训练好的enbedding权重矩阵再进行微调，这是多模态知识图谱常见做法。
        self.img_entity_embeddings = nn.Embedding.from_pretrained(img, freeze=False) #freeze=False，说明可以微调
        self.img_relation_embeddings = nn.Embedding(#构造图像模态的关系embdding
            2 * len(args.relation2id),
            args.r_dim,
            padding_idx=None
        )
        nn.init.xavier_normal_(self.img_relation_embeddings.weight)
        # 文本同理
        self.txt_entity_embeddings = nn.Embedding.from_pretrained(txt, freeze=False)
        self.txt_relation_embeddings = nn.Embedding(
            2 * len(args.relation2id),
            args.r_dim,
            padding_idx=None
        )
        nn.init.xavier_normal_(self.txt_relation_embeddings.weight)

        # 评分函数
        self.dim = args.dim
        self.img_dim = self.img_entity_embeddings.weight.data.shape[1]
        self.txt_dim = self.txt_entity_embeddings.weight.data.shape[1]
        self.fuse_out_dim = self.dim

        # 多模态MOE层 让模型自动学习不同“专家”子网络的加权组合
        self.visual_moe = MoEAdaptorLayer(n_exps=args.n_exp, layers=[self.img_dim, self.img_dim])
        self.text_moe = MoEAdaptorLayer(n_exps=args.n_exp, layers=[self.txt_dim, self.txt_dim])
        self.structure_moe = MoEAdaptorLayer(n_exps=args.n_exp, layers=[self.dim, self.dim])
        self.mm_moe = MoEAdaptorLayer(n_exps=args.n_exp, layers=[self.fuse_out_dim, self.fuse_out_dim])
        # Multi-modal fusion layers 多模态融合层
        self.fuse_e = ModalFusionLayer(in_dim=args.dim, out_dim=self.fuse_out_dim, multi=2, img_dim=self.img_dim,
                                       txt_dim=self.txt_dim)
        self.fuse_r = ModalFusionLayer(in_dim=args.r_dim, out_dim=self.fuse_out_dim, multi=2, img_dim=args.r_dim,
                                       txt_dim=args.r_dim)

        self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.bceloss=nn.BCELoss()

    def forward(self, batch_inputs, adj):
        #batch_inputs为triple三元组[[h1,r1,t1],[h2,r2,t2],...]
        head = batch_inputs[:, 0]  #[b] 拿到头实体
        relation = batch_inputs[:, 1] #[b] 拿到关系实体

        #取三模态 embedding
        stru_head = self.entity_embeddings(head) #[ b,dim] 实体信息
        img_head=self.img_entity_embeddings(head) #[b,dim]  图相信息
        txt_head=self.txt_entity_embeddings(head) #[b,dim]  文本信息
        # 进入 MoE
        e_embed, disen_str, atten_s = self.structure_moe(stru_head )  #结构模态moe
        e_img_embed, disen_img, atten_i = self.visual_moe(img_head)   #图像模态moe
        e_txt_embed, disen_txt, atten_t = self.text_moe(txt_head)      #文本模态moe

        r_embed = self.relation_embeddings(relation)
        r_img_embed,r_txt_embed=r_embed,r_embed

        # 2、融合(三模态堆叠，经过dense变成三个gate并进行加权合并)
        # [b,dim] [b,3]
        e_mm_embed, attn_f = self.fuse_e(e_embed, e_img_embed, e_txt_embed)   #融合模态moe
        r_mm_embed, _ = self.fuse_r(r_embed, r_img_embed, r_txt_embed)

        # 3、各模态预测得分  计算四种预测得分
        # e_embed [b,dim],r_embed=[b,dim]
        pred_s = e_embed*r_embed    #[b,dim]
        pred_i = e_img_embed* r_img_embed
        pred_d = e_txt_embed* r_txt_embed
        pred_mm = e_mm_embed* r_mm_embed

        # 4、预测所有类
        # self.entity_embeddings.weight=[num_entities, dim]
        # all_s [num,dim]
        all_s=self.entity_embeddings.weight  #[all_entity,dim]  -> [dim,all_entity]
        all_v=self.img_entity_embeddings.weight
        all_t = self.txt_entity_embeddings.weight
        all_f, _ = self.fuse_e(all_s, all_v, all_t)  # [num,dim]
          #与所有实体做矩阵乘法
        pred_s = torch.mm(pred_s, all_s.transpose(1, 0))  # [b,allentity]
        pred_i = torch.mm(pred_i, all_v.transpose(1, 0))  # [b,num]
        pred_d = torch.mm(pred_d, all_t.transpose(1, 0))  # [b,num]
        pred_mm = torch.mm(pred_mm, all_f.transpose(1, 0))  # [b,num]
        #四个模态  图相 文本 结构 融合 这是多标签二分类形式
        pred_s = torch.sigmoid(pred_s)  # [b,num]
        pred_i = torch.sigmoid(pred_i)  # [b,num]
        pred_d = torch.sigmoid(pred_d)  # [b,num]
        pred_mm = torch.sigmoid(pred_mm)  # [b,num]
        if not self.training:
            return [pred_s, pred_i, pred_d, pred_mm], [atten_s, atten_i, atten_t, attn_f]
        else:
            return [pred_s, pred_i, pred_d, pred_mm], [disen_str, disen_img, disen_txt]

    def get_batch_embeddings(self, batch_inputs):
        head = batch_inputs[:, 0]
        _, disen_str, _ = self.structure_moe(self.entity_embeddings(head))
        _, disen_img, _ = self.visual_moe(self.img_entity_embeddings(head))
        _, disen_txt, _ = self.text_moe(self.txt_entity_embeddings(head))
        return [disen_str, disen_img, disen_txt]

    def loss_func(self, output, target):  #四个模态损失相加。
        loss_s = self.bceloss(output[0], target)
        loss_i = self.bceloss(output[1], target)
        loss_d = self.bceloss(output[2], target)
        loss_mm = self.bceloss(output[3], target)
        return loss_s + loss_i + loss_d + loss_mm







