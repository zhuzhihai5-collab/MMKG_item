import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
CUDA = torch.cuda.is_available()

class LowRankerFusion(nn.Module):
    def __init__(self,dim):
        super(LowRankerFusion,self).__init__()
        self.struct_fc=nn.Linear(dim,dim)
        self.txt_fc=nn.Linear(dim,dim)
        self.img_fc=nn.Linear(dim,dim)
        self.all=nn.Linear(dim,dim)
    def forward(self,s_emb,v_emb,t_emb):
        fusion_zy=1.0
        s=self.struct_fc(s_emb)
        t=self.txt_fc(t_emb)
        v=self.img_fc(v_emb)
        fusion_zy=fusion_zy*s*t*v
        fused=self.all(fusion_zy)
        return fused

class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)


class LayerNormWithResidual(nn.Module):
    def __init__(self, input_dim,dropout_rate=0.1):
        super(LayerNormWithResidual, self).__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim*2)
        self.linear = nn.Linear(input_dim*2, input_dim*2)
        self.linear2 = nn.Linear(input_dim*2, input_dim)
        self.dropout=nn.Dropout(dropout_rate)
    def forward(self, x):
        residual = x
        x = self.linear(x)  # 线性变换
        tmp=self.batch_norm(F.leaky_relu(x) + residual)
        return  F.leaky_relu(self.linear2(tmp))# 添加残差连接
    def forward2(self, x):
        x = self.linear(x)  # 线性变换
        residual=x
        x = self.linear2(x)
        return self.batch_norm(self.dropout(x) + residual)  # 添加残差连接

class ConvKBLayer(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob, alpha_leaky):
        super(ConvKBLayer, self).__init__()

        self.conv_layer = nn.Conv2d(in_channels, out_channels, (1, input_seq_len))
        self.dropout = nn.Dropout(drop_prob)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear(input_dim * out_channels, 1)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)

    def forward(self, conv_input):
        batch_size, length, dim = conv_input.size()
        conv_input = conv_input.transpose(1, 2)
        conv_input = conv_input.unsqueeze(1)
        out_conv = self.dropout(self.non_linearity(self.conv_layer(conv_input)))
        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)
        return output


class MutanLayer(nn.Module):
    def __init__(self, dim, multi):
        super(MutanLayer, self).__init__()

        self.dim = dim
        self.multi = multi

        modal1 = []
        for i in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(dim, dim)
            modal1.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal1_layers = nn.ModuleList(modal1)

        modal2 = []
        for i in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(dim, dim)
            modal2.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal2_layers = nn.ModuleList(modal2)

        modal3 = []
        for i in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(dim, dim)
            modal3.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal3_layers = nn.ModuleList(modal3)

    def forward(self, modal1_emb, modal2_emb, modal3_emb):
        bs = modal1_emb.size(0)
        x_mm = []
        for i in range(self.multi):
            x_modal1 = self.modal1_layers[i](modal1_emb)
            x_modal2 = self.modal2_layers[i](modal2_emb)
            x_modal3 = self.modal3_layers[i](modal3_emb)
            x_mm.append(torch.mul(torch.mul(x_modal1, x_modal2), x_modal3))
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(bs, self.dim)
        x_mm = torch.relu(x_mm)
        return x_mm


class TuckERLayer(nn.Module):
    def __init__(self, dim, r_dim):
        super(TuckERLayer, self).__init__()
        
        self.W = nn.Parameter(torch.rand(r_dim, dim, dim))
        nn.init.xavier_uniform_(self.W.data)
        self.bn0 = nn.BatchNorm1d(dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.input_drop = nn.Dropout(0.3)
        self.hidden_drop = nn.Dropout(0.4)
        self.out_drop = nn.Dropout(0.5)

    def forward(self, e_embed, r_embed):
        #[b,dim]
        x = self.bn0(e_embed)
        x = self.input_drop(x)
        x = x.view(-1, 1, x.size(1)) #[b,1,dim]
        
        r = torch.mm(r_embed, self.W.view(r_embed.size(1), -1))#[b,dim]*[dim,dim*dim]=[b,dim*dim]
        r = r.view(-1, x.size(2), x.size(2)) #[b,dim,dim]
        r = self.hidden_drop(r)
       
        x = torch.bmm(x, r) #[b,1,dim] * [b,dim,dim]=[b,1,dim]
        x = x.view(-1, x.size(2)) #[b,dim]
        x = self.bn1(x)
        x = self.out_drop(x)
        return x #[b,dim]


class ConvELayer(nn.Module):
    def __init__(self, dim, out_channels, kernel_size, k_h, k_w):
        super(ConvELayer, self).__init__()

        self.input_drop = nn.Dropout(0.2)
        self.conv_drop = nn.Dropout2d(0.2)
        self.hidden_drop = nn.Dropout(0.2)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm1d(dim)

        self.conv = torch.nn.Conv2d(1, out_channels=out_channels, kernel_size=(kernel_size, kernel_size),
                                    stride=1, padding=0, bias=True)
        # assert k_h * k_w == dim
        flat_sz_h = int(2*k_w) - kernel_size + 1
        flat_sz_w = k_h - kernel_size + 1
        self.flat_sz = flat_sz_h * flat_sz_w * out_channels
        self.fc = nn.Linear(self.flat_sz, dim, bias=True)

    def forward(self, conv_input):
        x = self.bn0(conv_input)
        x = self.input_drop(x)
        x = self.conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x



