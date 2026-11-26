import torch
import numpy as np
from collections import defaultdict

#Corpus          # 抽象基类（框架）   ConvECorpus用于 ConvE 类模型（多标签训练）     ConvKBCorpus 用于 ConvKB 类模型（负采样训练）
class Corpus:
    def __init__(self, args, train_data, val_data, test_data, entity2id, relation2id):
        self.device = args.device
        self.train_triples = train_data[0] #这三行就是把语料拿过来了，把数据放到一个类里
        self.val_triples = val_data[0]
        self.test_triples = test_data[0]
        self.max_batch_num = 1

        adj_indices = torch.LongTensor([train_data[1][0], train_data[1][1]])
        adj_values = torch.LongTensor([train_data[1][2]])
        self.train_adj_matrix = (adj_indices, adj_values)

        self.entity2id = {k: v for k, v in entity2id.items()}
        self.id2entity = {v: k for k, v in entity2id.items()}
        self.relation2id = {k: v for k, v in relation2id.items()}
        self.id2relation = {v: k for k, v in relation2id.items()}
        self.batch_size = args.batch_size

    def shuffle(self):
        raise NotImplementedError

    def get_batch(self, batch_num):
        raise NotImplementedError

    def get_validation_pred(self, model, split='test'):
        raise NotImplementedError

'''
因为在训练的时候给每个关系生成了反向的关系-r，并且把原有的三元组h, r, t变成了两个三元组，
一个还是h, r, t，另一个是t, -r, h，这样两个三元组都进行尾实体的预测，
训练和测试都是这样，相当于正反两个方向都进行了，KGC模型中这种做法很常见
'''
class ConvECorpus(Corpus):
    def __init__(self, args, train_data, val_data, test_data, entity2id, relation2id):
        super(ConvECorpus, self).__init__(args, train_data, val_data, test_data, entity2id, relation2id)
        rel_num = len(relation2id)
        for k, v in relation2id.items():
            self.relation2id[k+'_reverse'] = v+rel_num
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        # 以下代码段的顺序严格，不允许更改
        sr2o = {} #subject+relation->object(sr2o)
        for (head, relation, tail) in self.train_triples:   #该代码构建了从 (实体, 关系) 到所有正确尾实体的映射，同时构造反向关系，用于统一处理头实体和尾实体预测任务，并支持负采样与过滤评估。
            if (head, relation) not in sr2o.keys():
                sr2o[(head, relation)] = set()
            if (tail, relation+rel_num) not in sr2o.keys():
                sr2o[(tail, relation+rel_num)] = set()
            sr2o[(head, relation)].add(tail)
            sr2o[(tail, relation+rel_num)].add(head)

        self.triples = {}
        self.train_indices = [{'triple': (a, b, -1), 'label': list(sr2o[(a, b)])}
                              for (a, b), c in sr2o.items()]
        self.triples['train'] = [{'triple': (head, relation, -1), 'label': list(sr2o[(head, relation)])}
                                 for (head, relation), tail in sr2o.items()]

        if len(self.train_indices) % self.batch_size == 0:
            self.max_batch_num = len(self.train_indices) // self.batch_size
        else:
            self.max_batch_num = len(self.train_indices) // self.batch_size + 1

        for (head, relation, tail) in self.val_triples:
            if (head, relation) not in sr2o.keys():
                sr2o[(head, relation)] = set()
            if (tail, relation+rel_num) not in sr2o.keys():
                sr2o[(tail, relation+rel_num)] = set()
            sr2o[(head, relation)].add(tail)
            sr2o[(tail, relation+rel_num)].add(head)

        for (head, relation, tail) in self.test_triples:
            if (head, relation) not in sr2o.keys():
                sr2o[(head, relation)] = set()
            if (tail, relation+rel_num) not in sr2o.keys():
                sr2o[(tail, relation+rel_num)] = set()
            sr2o[(head, relation)].add(tail) #说明同一个e1,r，可以有多个e2
            sr2o[(tail, relation+rel_num)].add(head)
        # 这里无论是val还是test，都进行了所有尾巴的预测（包括训练集也重新验证了一次）
        self.val_head_indices = [{'triple': (tail, relation + rel_num, head), 'label': list(sr2o[(tail, relation + rel_num)])}
                                 for (head, relation, tail) in self.val_triples]
        self.val_tail_indices = [{'triple': (head, relation, tail), 'label': list(sr2o[(head, relation)])}
                                 for (head, relation, tail) in self.val_triples]
        self.test_head_indices = [{'triple': (tail, relation + rel_num, head), 'label': list(sr2o[(tail, relation + rel_num)])}
                                 for (head, relation, tail) in self.test_triples]
        self.test_tail_indices = [{'triple': (head, relation, tail), 'label': list(sr2o[(head, relation)])}
                                 for (head, relation, tail) in self.test_triples]

        self.hits1 = 0
        self.hits3 = 0
        self.hits10 = 0
        self.hits100 = 0
        self.mean_rank = 0
    def read_batch(self, batch):
        triple, label = [_.to(self.device) for _ in batch]
        return triple, label

    def shuffle(self):
        np.random.shuffle(self.train_indices)

    def get_batch(self, batch_num):
        if (batch_num + 1) * self.batch_size <= len(self.train_indices):
            batch = self.train_indices[batch_num * self.batch_size: (batch_num+1) * self.batch_size]
        else:
            batch = self.train_indices[batch_num * self.batch_size:]
        batch_indices = torch.LongTensor([indice['triple'] for indice in batch])
        label = [np.int32(indice['label']) for indice in batch]
        #y.shape=[bs,all_label] 初始为0，列代表标签
        y = np.zeros((len(batch), len(self.entity2id)), dtype=np.float32)
        for idx in range(len(label)):
            for l in label[idx]:
                y[idx][l] = 1.0
        y = 0.9 * y + (1.0 / len(self.entity2id))
        batch_values = torch.FloatTensor(y)
        return batch_indices, batch_values #[bs,all_label]

    def get_validation_pred(self, model, split='test'):  #这个函数就是将训练出来的数据进行评价
        ranks_head, ranks_tail = [], []
        reciprocal_ranks_head, reciprocal_ranks_tail = [], []
        hits_at_100_head, hits_at_100_tail = 0, 0
        hits_at_10_head, hits_at_10_tail = 0, 0
        hits_at_3_head, hits_at_3_tail = 0, 0
        hits_at_1_head, hits_at_1_tail = 0, 0

        rel_pred_dict = defaultdict(list)
        att_s = []
        att_i = []
        att_t = []
        att_mm = []

        if split == 'val':
            head_indices = self.val_head_indices
            tail_indices = self.val_tail_indices
        else:
            head_indices = self.test_head_indices #[{triple:三元组，label:标签},,,,,]
            tail_indices = self.test_tail_indices #[{triple:三元组，label:标签},,,,,]

        if len(head_indices) % self.batch_size == 0:
            max_batch_num = len(head_indices) // self.batch_size
        else:
            max_batch_num = len(head_indices) // self.batch_size + 1

        for batch_num in range(max_batch_num):
            # [{triple:三元组，label:标签},,,,,]
            if (batch_num + 1) * self.batch_size <= len(head_indices):
                head_batch = head_indices[batch_num * self.batch_size: (batch_num + 1) * self.batch_size]
                tail_batch = tail_indices[batch_num * self.batch_size: (batch_num + 1) * self.batch_size]
            else:
                head_batch = head_indices[batch_num * self.batch_size:]
                tail_batch = tail_indices[batch_num * self.batch_size:]

            head_batch_indices = torch.LongTensor([indice['triple'] for indice in head_batch]) # [b,3]  [[e1,r,e2],,,,,]
            head_batch_indices = head_batch_indices.to(self.device)
            rel_ids = head_batch_indices[:, 1] #[b,]
            pred, attention = model.forward(head_batch_indices,self.train_adj_matrix) #[4,b,all]
            #————————————————————————————————————————————————————————————————————————————————
            for i in range(pred[0].shape[0]):
                h, r, t = head_batch_indices[i][0].item(), head_batch_indices[i][1].item(), head_batch_indices[i][2].item()
                atts = attention[0][i]
                atti = attention[1][i]
                attt = attention[2][i]
                attmm = attention[3][i]
                att_s.append((h, r, t, atts))
                att_i.append((h, r, t, atti))
                att_t.append((h, r, t, attt))
                att_mm.append((h, r, t, attmm))

            # 先获取当前batch的预估分，再获取所有数据的label
            pred = (pred[0] + pred[1] + pred[2] + pred[3]) / 4.0 #[b,all] #triple正儿八经的bs匹配(e1,r,e2)，而label可能有多个标签(e2,e3,e4...)
            label = [np.int32(indice['label']) for indice in head_batch] #[b,lenx]
            y = np.zeros((len(head_batch), len(self.entity2id)), dtype=np.float32) #[b,all]
            for idx in range(len(label)):
                for l in label[idx]:
                    y[idx][l] = 1.0
            y = torch.FloatTensor(y).to(self.device) #[b,all]
            # 相当于把当前batch的label设为1，而不是全部，因为这个y里面是所有的label
            target = head_batch_indices[:, 2] #[1024] 1v1
            b_range = torch.arange(pred.shape[0], device=self.device)
            target_pred = pred[b_range, target] #[1024] 1v1 当前批次的label预估分
            # 先把所有label对应的预估分设为0，然后把当前批次的label预估分加进去
            pred = torch.where(y.byte(), torch.zeros_like(pred), pred) #[1024,all]
            pred[b_range, target] = target_pred
            pred = pred.cpu().numpy()
            target = target.cpu().numpy()

            # pred.shape[0] = batch_size
            for i in range(pred.shape[0]):
                scores = pred[i]
                tar = target[i]
                tar_scr = scores[tar]
                #从scores中删除目标标签对应的预测得分tar_scr。
                scores = np.delete(scores, tar) #这一步可能是为了模拟测试场景下目标标签未知的情况，即删除真实标签分数，以防其干扰排序
                rand = np.random.randint(scores.shape[0])
                scores = np.insert(scores, rand, tar_scr) #删了，又随机插回来，目标标签的预测得分保留在随机位置
                sorted_indices = np.argsort(-scores, kind='stable') #对scores按从高到低排序，并返回排序后的索引，得分越高，索引越靠前
                #由于位置偏差，主要还是稳定排序本身的问题，如果值一样，他的排名就一直靠前，故引入随机噪声，每次位置不是稳定的，但是值唯一的话就无所谓了
                # higher is better
                ranks_head.append(np.where(sorted_indices == rand)[0][0]+1) #用于从数组中找出满足条件的元素的位置,相当于找出真实标签的排名，并将排名添加
                reciprocal_ranks_head.append(1.0 / ranks_head[-1]) #将其排名的倒数添加
                rel_pred_dict[rel_ids[i].item()].append(ranks_head[-1])

            tail_batch_indices = torch.LongTensor([indice['triple'] for indice in tail_batch])
            tail_batch_indices = tail_batch_indices.to(self.device)
            rel_ids = tail_batch_indices[:, 1]
            pred, attention = model.forward(tail_batch_indices,self.train_adj_matrix)
            for i in range(pred[0].shape[0]):
                h, r, t = tail_batch_indices[i][0].item(), tail_batch_indices[i][1].item(), tail_batch_indices[i][2].item()
                atts = attention[0][i]
                atti = attention[1][i]
                attt = attention[2][i]
                attmm = attention[3][i]
                att_s.append((h, r, t, atts))
                att_i.append((h, r, t, atti))
                att_t.append((h, r, t, attt))
                att_mm.append((h, r, t, attmm))
            pred = (pred[0] + pred[1] + pred[2] + pred[3]) / 4.0
            label = [np.int32(indice['label']) for indice in tail_batch]
            y = np.zeros((len(tail_batch), len(self.entity2id)), dtype=np.float32)
            for idx in range(len(label)):
                for l in label[idx]:
                    y[idx][l] = 1.0
            y = torch.FloatTensor(y).to(self.device)
            target = tail_batch_indices[:, 2]
            b_range = torch.arange(pred.shape[0], device=self.device)
            target_pred = pred[b_range, target]
            pred = torch.where(y.byte(), torch.zeros_like(pred), pred)
            pred[b_range, target] = target_pred
            pred = pred.cpu().numpy()
            target = target.cpu().numpy()
            for i in range(pred.shape[0]):
                scores = pred[i]
                tar = target[i]
                tar_scr = scores[tar]
                scores = np.delete(scores, tar)
                rand = np.random.randint(scores.shape[0])
                scores = np.insert(scores, rand, tar_scr)
                sorted_indices = np.argsort(-scores, kind='stable')
                ranks_tail.append(np.where(sorted_indices == rand)[0][0] + 1)
                reciprocal_ranks_tail.append(1.0 / ranks_tail[-1])
                rel_pred_dict[rel_ids[i].item()].append(ranks_head[-1])
        # ————————————————————————————————————————————————————————————————————————————————
        #ranks_head：预测头的排名
        for i in range(len(ranks_head)): #记录预测对应标签的得分排名，topK的命中次数
            if ranks_head[i] <= 100:
                hits_at_100_head += 1
            if ranks_head[i] <= 10:
                hits_at_10_head += 1
            if ranks_head[i] <= 3:
                hits_at_3_head += 1
            if ranks_head[i] == 1:
                hits_at_1_head += 1

        for i in range(len(ranks_tail)):
            if ranks_tail[i] <= 100:
                hits_at_100_tail += 1
            if ranks_tail[i] <= 10:
                hits_at_10_tail += 1
            if ranks_tail[i] <= 3:
                hits_at_3_tail += 1
            if ranks_tail[i] == 1:
                hits_at_1_tail += 1
        assert len(ranks_head) == len(reciprocal_ranks_head)
        assert len(ranks_tail) == len(reciprocal_ranks_tail)

        # ————————————————————————————————————————————————————————————————————————————————
        #评价头
        hits_100_head = hits_at_100_head / len(ranks_head) #前100命中率
        hits_10_head = hits_at_10_head / len(ranks_head)
        hits_3_head = hits_at_3_head / len(ranks_head)
        hits_1_head = hits_at_1_head / len(ranks_head)
        mean_rank_head = sum(ranks_head) / len(ranks_head) #平均排名
        mean_reciprocal_rank_head = sum(reciprocal_ranks_head) / len(reciprocal_ranks_head) #平均倒数排名
        #评价尾
        hits_100_tail = hits_at_100_tail / len(ranks_tail)
        hits_10_tail = hits_at_10_tail / len(ranks_tail)
        hits_3_tail = hits_at_3_tail / len(ranks_tail)
        hits_1_tail = hits_at_1_tail / len(ranks_tail)
        mean_rank_tail = sum(ranks_tail) / len(ranks_tail)
        mean_reciprocal_rank_tail = sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail)
        # ————————————————————————————————————————————————————————————————————————————————
        #综合
        hits_100 = (hits_100_head + hits_100_tail) / 2
        hits_10 = (hits_10_head + hits_10_tail) / 2
        hits_3 = (hits_3_head + hits_3_tail) / 2
        hits_1 = (hits_1_head + hits_1_tail) / 2
        mean_rank = (mean_rank_head + mean_rank_tail) / 2
        mean_reciprocal_rank = (mean_reciprocal_rank_head + mean_reciprocal_rank_tail) / 2

        metrics = {
            "Hits@100": hits_100,
            "Hits@10": hits_10,
            "Hits@3": hits_3,
            "Hits@1": hits_1,
            "MR": mean_rank,
            "MRR": mean_reciprocal_rank
        }

        return metrics, [att_s, att_i, att_t, att_mm]
    
    def get_validation_pred_signle(self, model, split='test', index=0):#和上面函数类似，但只使用某一个子模型的输出（比如只看图像模态）用于分析不同模态贡献。
        ranks_head, ranks_tail = [], []
        reciprocal_ranks_head, reciprocal_ranks_tail = [], []
        hits_at_100_head, hits_at_100_tail = 0, 0
        hits_at_10_head, hits_at_10_tail = 0, 0
        hits_at_3_head, hits_at_3_tail = 0, 0
        hits_at_1_head, hits_at_1_tail = 0, 0
        rel_pred_dict = defaultdict(list)

        if split == 'val':
            head_indices = self.val_head_indices
            tail_indices = self.val_tail_indices
        else:
            head_indices = self.test_head_indices
            tail_indices = self.test_tail_indices

        if len(head_indices) % self.batch_size == 0:
            max_batch_num = len(head_indices) // self.batch_size
        else:
            max_batch_num = len(head_indices) // self.batch_size + 1
        for batch_num in range(max_batch_num):
            if (batch_num + 1) * self.batch_size <= len(head_indices):
                head_batch = head_indices[batch_num * self.batch_size: (batch_num + 1) * self.batch_size]
                tail_batch = tail_indices[batch_num * self.batch_size: (batch_num + 1) * self.batch_size]
            else:
                head_batch = head_indices[batch_num * self.batch_size:]
                tail_batch = tail_indices[batch_num * self.batch_size:]

            head_batch_indices = torch.LongTensor([indice['triple'] for indice in head_batch])
            head_batch_indices = head_batch_indices.to(self.device)
            rel_ids = head_batch_indices[:, 1]
            pred ,_= model.forward(head_batch_indices,self.train_adj_matrix)
            pred=pred[index]
            label = [np.int32(indice['label']) for indice in head_batch]
            y = np.zeros((len(head_batch), len(self.entity2id)), dtype=np.float32)
            for idx in range(len(label)):
                for l in label[idx]:
                    y[idx][l] = 1.0
            y = torch.FloatTensor(y).to(self.device)
            target = head_batch_indices[:, 2]
            # b_range = torch.arange(len(pred), device=self.device)
            b_range = torch.arange(pred.shape[0], device=self.device)
            target_pred = pred[b_range, target]
            pred = torch.where(y.byte(), torch.zeros_like(pred), pred)
            pred[b_range, target] = target_pred
            pred = pred.cpu().numpy()
            target = target.cpu().numpy()
            for i in range(pred.shape[0]):
                scores = pred[i]
                tar = target[i]
                tar_scr = scores[tar]
                scores = np.delete(scores, tar)
                rand = np.random.randint(scores.shape[0])
                scores = np.insert(scores, rand, tar_scr)
                sorted_indices = np.argsort(-scores, kind='stable')
                # higher is better
                ranks_head.append(np.where(sorted_indices == rand)[0][0]+1)
                reciprocal_ranks_head.append(1.0 / ranks_head[-1])
                rel_pred_dict[rel_ids[i].item()].append(ranks_head[-1])

            tail_batch_indices = torch.LongTensor([indice['triple'] for indice in tail_batch])
            tail_batch_indices = tail_batch_indices.to(self.device)
            rel_ids = tail_batch_indices[:, 1]
            pred = model.forward(tail_batch_indices,self.train_adj_matrix)[0][index]
            label = [np.int32(indice['label']) for indice in tail_batch]
            y = np.zeros((len(tail_batch), len(self.entity2id)), dtype=np.float32)
            for idx in range(len(label)):
                for l in label[idx]:
                    y[idx][l] = 1.0
            y = torch.FloatTensor(y).to(self.device)
            target = tail_batch_indices[:, 2]
            b_range = torch.arange(pred.shape[0], device=self.device)
            target_pred = pred[b_range, target]
            pred = torch.where(y.byte(), torch.zeros_like(pred), pred)
            pred[b_range, target] = target_pred
            pred = pred.cpu().numpy()
            target = target.cpu().numpy()
            for i in range(pred.shape[0]):
                scores = pred[i]
                tar = target[i]
                tar_scr = scores[tar]
                scores = np.delete(scores, tar)
                rand = np.random.randint(scores.shape[0])
                scores = np.insert(scores, rand, tar_scr)
                sorted_indices = np.argsort(-scores, kind='stable')
                ranks_tail.append(np.where(sorted_indices == rand)[0][0] + 1)
                reciprocal_ranks_tail.append(1.0 / ranks_tail[-1])
                rel_pred_dict[rel_ids[i].item()].append(ranks_head[-1])

        for i in range(len(ranks_head)):
            if ranks_head[i] <= 100:
                hits_at_100_head += 1
            if ranks_head[i] <= 10:
                hits_at_10_head += 1
            if ranks_head[i] <= 3:
                hits_at_3_head += 1
            if ranks_head[i] == 1:
                hits_at_1_head += 1

        for i in range(len(ranks_tail)):
            if ranks_tail[i] <= 100:
                hits_at_100_tail += 1
            if ranks_tail[i] <= 10:
                hits_at_10_tail += 1
            if ranks_tail[i] <= 3:
                hits_at_3_tail += 1
            if ranks_tail[i] == 1:
                hits_at_1_tail += 1

        assert len(ranks_head) == len(reciprocal_ranks_head)
        assert len(ranks_tail) == len(reciprocal_ranks_tail)

        hits_100_head = hits_at_100_head / len(ranks_head)
        hits_10_head = hits_at_10_head / len(ranks_head)
        hits_3_head = hits_at_3_head / len(ranks_head)
        hits_1_head = hits_at_1_head / len(ranks_head)
        mean_rank_head = sum(ranks_head) / len(ranks_head)
        mean_reciprocal_rank_head = sum(reciprocal_ranks_head) / len(reciprocal_ranks_head)

        hits_100_tail = hits_at_100_tail / len(ranks_tail)
        hits_10_tail = hits_at_10_tail / len(ranks_tail)
        hits_3_tail = hits_at_3_tail / len(ranks_tail)
        hits_1_tail = hits_at_1_tail / len(ranks_tail)
        mean_rank_tail = sum(ranks_tail) / len(ranks_tail)
        mean_reciprocal_rank_tail = sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail)

        hits_100 = (hits_at_100_head / len(ranks_head) + hits_at_100_tail / len(ranks_tail)) / 2
        hits_10 = (hits_at_10_head / len(ranks_head) + hits_at_10_tail / len(ranks_tail)) / 2
        hits_3 = (hits_at_3_head / len(ranks_head) + hits_at_3_tail / len(ranks_tail)) / 2
        hits_1 = (hits_at_1_head / len(ranks_head) + hits_at_1_tail / len(ranks_tail)) / 2
        mean_rank = (sum(ranks_head) / len(ranks_head) + sum(ranks_tail) / len(ranks_tail)) / 2
        mean_reciprocal_rank = (sum(reciprocal_ranks_head) / len(reciprocal_ranks_head) + sum(
            reciprocal_ranks_tail) / len(reciprocal_ranks_tail)) / 2

        metrics = {
            "Hits@100": hits_100,
            "Hits@10": hits_10,
            "Hits@3": hits_3,
            "Hits@1": hits_1,
            "MR": mean_rank,
            "MRR": mean_reciprocal_rank
        }

        return metrics, rel_pred_dict


class ConvKBCorpus(Corpus):
    def __init__(self, args, train_data, val_data, test_data, entity2id, relation2id):
        super(ConvKBCorpus, self).__init__(args, train_data, val_data, test_data, entity2id, relation2id)
        self.neg_num = args.neg_num
        if len(self.train_triples) % self.batch_size == 0:
            self.max_batch_num = len(self.train_triples) // self.batch_size
        else:
            self.max_batch_num = len(self.train_triples) // self.batch_size + 1

        self.train_indices = np.array(self.train_triples).astype(np.int32)
        self.train_values = np.array([[1]] * len(self.train_triples)).astype(np.float32)
        self.val_indices = np.array(self.val_triples).astype(np.int32)
        self.val_values = np.array([[1]] * len(self.val_triples)).astype(np.float32)
        self.test_indices = np.array(self.test_triples).astype(np.int32)
        self.test_values = np.array([[1]] * len(self.test_triples)).astype(np.float32)

        self.unique_entities = [entity2id[i] for i in train_data[2]]
        self.all_triples = {j: i for i, j in enumerate(self.train_triples + self.val_triples + self.test_triples)}

        self.batch_indices = np.empty((self.batch_size * (self.neg_num + 1), 3)).astype(np.int32)
        self.batch_values = np.empty((self.batch_size * (self.neg_num + 1), 1)).astype(np.float32)

    def shuffle(self):
        np.random.shuffle(self.train_indices)

    def get_batch(self, batch_num):
        if (batch_num + 1) * self.batch_size <= len(self.train_indices):
            self.batch_indices = np.empty((self.batch_size * (self.neg_num + 1), 3)).astype(np.int32)
            self.batch_values = np.empty((self.batch_size * (self.neg_num + 1), 1)).astype(np.float32)

            indices = range(self.batch_size * batch_num, self.batch_size * (batch_num + 1))
            last_index = self.batch_size

        else:
            last_batch_size = len(self.train_indices) - self.batch_size * batch_num
            self.batch_indices = np.empty((last_batch_size * (self.neg_num + 1), 3)).astype(np.int32)
            self.batch_values = np.empty((last_batch_size * (self.neg_num + 1), 1)).astype(np.float32)

            indices = range(self.batch_size * batch_num, len(self.train_indices))
            last_index = last_batch_size

        self.batch_indices[:last_index, :] = self.train_indices[indices, :]
        self.batch_values[:last_index, :] = self.train_values[indices, :]
        random_entities = np.random.randint(0, len(self.entity2id), last_index * self.neg_num)
        self.batch_indices[last_index: (last_index * (self.neg_num + 1)), :] = np.tile(
            self.batch_indices[:last_index, :], (self.neg_num, 1))
        self.batch_values[last_index: (last_index * (self.neg_num + 1)), :] = np.tile(
            self.batch_values[:last_index, :], (self.neg_num, 1))

        for i in range(last_index):
            for j in range(self.neg_num // 2):
                current_index = i * (self.neg_num // 2) + j

                while(random_entities[current_index], self.batch_indices[last_index + current_index, 1],
                      self.batch_indices[last_index + current_index, 2]) in self.all_triples.keys():
                    random_entities[current_index] = np.random.randint(0, len(self.entity2id))

                self.batch_indices[last_index + current_index, 0] = random_entities[current_index]
                self.batch_values[last_index + current_index, :] = [-1]
            for j in range(self.neg_num // 2):
                current_index = last_index * (self.neg_num // 2) + i * (self.neg_num // 2) + j

                while (self.batch_indices[last_index + current_index, 0], self.batch_indices[last_index + current_index, 1],
                       random_entities[current_index]) in self.all_triples.keys():
                    random_entities[current_index] = np.random.randint(0, len(self.entity2id))

                self.batch_indices[last_index + current_index, 2] = random_entities[current_index]
                self.batch_values[last_index + current_index, :] = [-1]

        return self.batch_indices, self.batch_values

    def get_validation_pred(self, model, split='test'):
        ranks_head, ranks_tail = [], []
        reciprocal_ranks_head, reciprocal_ranks_tail = [], []
        hits_at_100_head, hits_at_100_tail = 0, 0
        hits_at_10_head, hits_at_10_tail = 0, 0
        hits_at_3_head, hits_at_3_tail = 0, 0
        hits_at_1_head, hits_at_1_tail = 0, 0
        entity_list = [i for i in self.entity2id.values()]
        if split == 'val':
            split_triples = np.array(self.val_triples).astype(np.int32)
        elif split == 'test':
            split_triples = np.array(self.test_triples).astype(np.int32)

        for i in range(split_triples.shape[0]):
            if split_triples[i, 0] not in self.unique_entities or split_triples[i, 2] not in self.unique_entities:
                continue
            x_head = np.tile(split_triples[i, :], (len(self.entity2id), 1))
            x_tail = np.tile(split_triples[i, :], (len(self.entity2id), 1))
            x_head[:, 0] = entity_list
            x_tail[:, 2] = entity_list

            last_index_head, last_index_tail = [], []
            for idx in range(len(x_head)):
                head = (x_head[idx][0], x_head[idx][1], x_head[idx][2])
                if head in self.all_triples.keys():
                    last_index_head.append(idx)

                tail = (x_tail[idx][0], x_tail[idx][1], x_tail[idx][2])
                if tail in self.all_triples.keys():
                    last_index_tail.append(idx)

            x_head = np.delete(x_head, last_index_head, axis=0)
            x_tail = np.delete(x_tail, last_index_tail, axis=0)
            rand_head = np.random.randint(x_head.shape[0])
            rand_tail = np.random.randint(x_tail.shape[0])
            x_head = np.insert(x_head, rand_head, split_triples[i], axis=0)
            x_tail = np.insert(x_tail, rand_tail, split_triples[i], axis=0)
            x_head = torch.LongTensor(x_head).to(self.device)
            x_tail = torch.LongTensor(x_tail).to(self.device)
            #scores_head = model.forward(x_head)
            scores_head = model.predict(x_head)
            sorted_scores_head, sorted_triples_head = torch.sort(scores_head.view(-1), dim=-1, descending=True)
            ranks_head.append(np.where(sorted_triples_head.cpu().numpy() == rand_head)[0][0]+1)
            reciprocal_ranks_head.append(1.0 / ranks_head[-1])
            #scores_tail = model.forward(x_tail)
            scores_tail = model.predict(x_tail)
            sorted_scores_tail, sorted_triples_tail = torch.sort(scores_tail.view(-1), dim=-1, descending=True)
            ranks_tail.append(np.where(sorted_triples_tail.cpu().numpy() == rand_tail)[0][0]+1)
            reciprocal_ranks_tail.append(1.0 / ranks_tail[-1])

        for i in range(len(ranks_head)):
            if ranks_head[i] <= 100:
                hits_at_100_head += 1
            if ranks_head[i] <= 10:
                hits_at_10_head += 1
            if ranks_head[i] <= 3:
                hits_at_3_head += 1
            if ranks_head[i] == 1:
                hits_at_1_head += 1

        for i in range(len(ranks_tail)):
            if ranks_tail[i] <= 100:
                hits_at_100_tail += 1
            if ranks_tail[i] <= 10:
                hits_at_10_tail += 1
            if ranks_tail[i] <= 3:
                hits_at_3_tail += 1
            if ranks_tail[i] == 1:
                hits_at_1_tail += 1

        assert len(ranks_head) == len(reciprocal_ranks_head)
        assert len(ranks_tail) == len(reciprocal_ranks_tail)

        hits_100_head = hits_at_100_head / len(ranks_head)
        hits_10_head = hits_at_10_head / len(ranks_head)
        hits_3_head = hits_at_3_head / len(ranks_head)
        hits_1_head = hits_at_1_head / len(ranks_head)
        mean_rank_head = sum(ranks_head) / len(ranks_head)
        mean_reciprocal_rank_head = sum(reciprocal_ranks_head) / len(reciprocal_ranks_head)

        hits_100_tail = hits_at_100_tail / len(ranks_tail)
        hits_10_tail = hits_at_10_tail / len(ranks_tail)
        hits_3_tail = hits_at_3_tail / len(ranks_tail)
        hits_1_tail = hits_at_1_tail / len(ranks_tail)
        mean_rank_tail = sum(ranks_tail) / len(ranks_tail)
        mean_reciprocal_rank_tail = sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail)

        hits_100 = (hits_at_100_head / len(ranks_head) + hits_at_100_tail / len(ranks_tail)) / 2
        hits_10 = (hits_at_10_head / len(ranks_head) + hits_at_10_tail / len(ranks_tail)) / 2
        hits_3 = (hits_at_3_head / len(ranks_head) + hits_at_3_tail / len(ranks_tail)) / 2
        hits_1 = (hits_at_1_head / len(ranks_head) + hits_at_1_tail / len(ranks_tail)) / 2
        mean_rank = (sum(ranks_head) / len(ranks_head) + sum(ranks_tail) / len(ranks_tail)) / 2
        mean_reciprocal_rank = (sum(reciprocal_ranks_head) / len(reciprocal_ranks_head) + sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail)) / 2

        metrics = {
            "Hits@100_head": hits_100_head,
            "Hits@10_head": hits_10_head,
            "Hits@3_head": hits_3_head,
            "Hits@1_head": hits_1_head,
            "Mean Rank_head": mean_rank_head,
            "Mean Reciprocal Rank_head": mean_reciprocal_rank_head,
            "Hits@100_tail": hits_100_tail, "Hits@10_tail": hits_10_tail, "Hits@3_tail": hits_3_tail, "Hits@1_tail": hits_1_tail,
            "Mean Rank_tail": mean_rank_tail, "Mean Reciprocal Rank_tail": mean_reciprocal_rank_tail,
            "Hits@100": hits_100, "Hits@10": hits_10, "Hits@3": hits_3, "Hits@1": hits_1,
            "Mean Rank": mean_rank, "Mean Reciprocal Rank": mean_reciprocal_rank}

        return metrics


