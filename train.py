import argparse
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models.Multi_MoE import *
from models.model import *
from utils.data_loader import *
from utils.data_util import load_data

####################################1、定义训练需要的参数########################################################
def parse_args():
    config_args = {
        #decoder部分
        'epochs': 2000,  # 2k  当模型把整个训练数据从头到尾全部训练一次，就叫做 1 个 epoch。
        'eval_freq': 100,  # 100  每 100 个 epoch 评估一次。
        'decoder_save_model':'./checkpoint/DB15K/trained_model.pth',
        #其他超参
        'dim': 256,
        'r_dim': 256,
        'layer_nums':4,
        'batch_size': 1024,
        'save': 1,
        'n_exp': 3,
        'mu': 0.0001,
        'img_dim': 256,
        'txt_dim': 256,
        'lr': 0.0005,
        'dropout_gat': 0.3,
        'dropout': 0.3,
        'cuda': 0,
        'weight_decay_gat': 1e-5,
        'weight_decay': 0,
        'seed': 10010,
        'model': 'RMoE',
        'num-layers': 3,
        'k_w': 16,
        'k_h': 16,
        'n_heads': 2,
        'dataset': 'DB15K',
        'encoder': 0,
        'image_features': 1,
        'text_features': 1,
        'patience': 5,
        'lr_reduce_freq': 500,
        'gamma': 1.0,
        'bias': 1,
        'neg_num': 2,
        'neg_num_gat': 2,
        'alpha': 0.2,
        'alpha_gat': 0.2,
        'out_channels': 32,
        'kernel_size': 3
    }

    parser = argparse.ArgumentParser()
    for param, val in config_args.items():
        parser.add_argument(f"--{param}", default=val, type=type(val))
    args = parser.parse_args()
    return args

#ps:返回参数字典
args = parse_args()
for k, v in list(vars(args).items()):
    print(str(k) + ':' + str(v))

#设置随机种子，确保结果重现
np.random.seed(args.seed)
torch.manual_seed(args.seed)

args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu' #准备GPU的
print(f'Using: {args.device}')
if args.cuda >= 0:
    torch.cuda.set_device(args.cuda)

####################################2、加载数据集#######################################################
#1、特征预处理（id转换，图文特征向量，数据集划分）
#train_data里面为（三元组列表，htr三个列表,唯一实体）
entity2id, relation2id, img_features, text_features, train_data, val_data, test_data = load_data(args.dataset)
print("Training data {:04d}".format(len(train_data[0])))

#2、语料装载到这个类中(将三元组的输入和输出加载进去，后期直接使用这个语料对象)
corpus = ConvECorpus(args, train_data, val_data, test_data, entity2id, relation2id) #将读取的数据加载到这个类中

#如果图文预处理的特征向量存在，则归一化处理一哈
if args.image_features:
    args.img = F.normalize(torch.Tensor(img_features), p=2, dim=1)
if args.text_features:
    args.desp = F.normalize(torch.Tensor(text_features), p=2, dim=1)
args.entity2id = entity2id
args.relation2id = relation2id


def train_decoder(args):
    ####################################3、定义模型及其组件#######################################################
    model = Multi_MoE(args)
    args.img_dim = model.img_dim
    args.txt_dim = model.txt_dim
    print(str(model))
    # weight_decay 是一种正则化技术，它通过在权重更新过程中加入一个额外的惩罚项来抑制权重过大，从而减少过拟合的风险。
    # lr_scheduler 则是用来控制学习率随训练进程的变化，比如降低学习率以帮助模型更好地收敛。
    # 优化器：优化模型参数
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.gamma) #学习率衰减器，每个epoch衰减一次
    #np.prod计算数组中所有元素的乘积
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f'Total number of parameters: {tot_params}')
    if args.cuda is not None and int(args.cuda) >= 0:
        model = model.to(args.device)  #将模型转移到GPU上

    #################################### 4、训练模型#######################################################


    # for 每个 epoch:
    #     打乱数据
    #     for 每个 batch:
    #         清梯度
    #         前向传播
    #         计算损失
    #         反向传播
    #         更新权重
    #     调整学习率
    t_total = time.time() #记录训练开始时间
    best_val_metrics = model.init_metric_dict()#初始化最佳指标
    best_test_metrics = model.init_metric_dict()#初始化最佳指标
    corpus.batch_size = args.batch_size #设置 batch 大小  每次喂多少数据给模型。
    training_range = tqdm(range(args.epochs))  #tqdm 进度条 用于显示：Epoch 3/500 这种进度条。
    for epoch in training_range:  #训练 args.epochs 轮
        model.train()  #进入训练模式  作用：1.开启 Dropout  2.BatchNorm 用当前 batch 3.允许梯度更新
        epoch_loss = []  #每个 epoch 的初始化
        corpus.shuffle()  #shuffle()  打乱训练数据。  防止模型记住顺序，提高泛化能力提高泛化能力
        for batch_num in range(corpus.max_batch_num):  #把一个 epoch 分成多个 batch 训练
            optimizer.zero_grad() #梯度清零
            train_indices, train_values=corpus.get_batch(batch_num)  #取一个 batch 数据  train_indices → 三元组索引  train_values → 标签
            train_indices = torch.LongTensor(train_indices) #转成 tensor（张量）
            if args.cuda is not None and int(args.cuda) >= 0:
                train_indices = train_indices.to(args.device) #转入GPU
                train_values = train_values.to(args.device)


            output, embeddings = model.forward(train_indices,corpus.train_adj_matrix) # 前向传播：至此准备工作在完成 进入训练阶段
            #[pred_s, pred_i, pred_d, pred_mm], [disen_str, disen_img, disen_txt]  前[b,num] 后[B, n_exps, dim]
            loss = model.loss_func(output,train_values) #计算损失
            loss.backward()#反向传播
            optimizer.step()#更新参数
            epoch_loss.append(loss.data.item())  #保存 loss    用于统计：这个 epoch 的平均损失
        lr_scheduler.step() #学习率调度

        #################################### 5、评估模型#######################################################

        #在训练模式下，Dropout会随机失活神经元，BatchNorm使用当前批次统计量并更新全局统计；而在评估模式下，Dropout被关闭，BatchNorm使用训练期间累积的统计量，从而保证推理结果稳定。同时在评估阶段通常配合 torch.no_grad()，避免梯度计算，提高效率，因为评估不需要反向传播。
        if (epoch + 1) % args.eval_freq == 0:
            print("Epoch数量 {:04d} , 平均损失 {:.4f} \n".format(
                epoch + 1, sum(epoch_loss) / len(epoch_loss)))
            training_range.set_postfix(loss="main: {:.5} ".format(sum(epoch_loss)))
            model.eval()
            print("==================================第",(epoch + 1) // args.eval_freq,"次评估========================================")
            with torch.no_grad():
                val_metrics = corpus.get_validation_pred(model, 'test')[0]
            if val_metrics['MRR'] > best_test_metrics['MRR']:
                best_test_metrics['MRR'] = val_metrics['MRR']
            if val_metrics['MR'] < best_test_metrics['MR']:
                best_test_metrics['MR'] = val_metrics['MR']
            if val_metrics['Hits@1'] > best_test_metrics['Hits@1']:
                best_test_metrics['Hits@1'] = val_metrics['Hits@1']
            if val_metrics['Hits@3'] > best_test_metrics['Hits@3']:
                best_test_metrics['Hits@3'] = val_metrics['Hits@3']
            if val_metrics['Hits@10'] > best_test_metrics['Hits@10']:
                best_test_metrics['Hits@10'] = val_metrics['Hits@10']
            if val_metrics['Hits@100'] > best_test_metrics['Hits@100']:
                best_test_metrics['Hits@100'] = val_metrics['Hits@100']
            print('\n'.join(['Epoch: {:04d}'.format(epoch + 1), model.format_metrics(val_metrics, 'test')]))
            print("\n\n")

    print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))

    # 如果没有best_test_metrics，默认将最后一次评估结果赋值给best_test_metrics
    if not best_test_metrics:
        model.eval()
        with torch.no_grad():
            best_test_metrics = corpus.get_validation_pred(model, 'test')


    print('\n'.join(['Val set results:', model.format_metrics(best_val_metrics, 'val')]))
    print('\n'.join(['Test set results:', model.format_metrics(best_test_metrics, 'test')]))
    print("\n\n\n")

    if args.save:
        Path(args.decoder_save_model).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.decoder_save_model)
        print('Saved model!')


if __name__ == '__main__':
    train_decoder(args)

