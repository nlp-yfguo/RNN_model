import torch
from torch import nn
import torch.nn.functional as F

class TestModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hsize=128, Hidden_state=32, bind_emb=True):
        super(TestModel, self).__init__()

        # self.zero_state = torch.nn.Parameter(torch.empty(32))
        self.wemb = nn.Embedding(vocab_size, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(emb_dim + Hidden_state, hsize),  # 32+32-->128
            nn.GELU(),
            nn.Linear(hsize, emb_dim)  # 128-->32
        )
        self.classifier = nn.Linear(emb_dim, vocab_size)  # 32-->vocab_size

        # 绑定词向量和分类器的权重
        if bind_emb:
            self.classifier.weight = self.wemb.weight

    def forward(self,input_batch,last_state):
        '''

        :param input_batch: [num_sentence, seql ]
        :param last_state: 上一词的隐状态，输入为1列  [num_sentence , 1 ,32]
        :return:
        '''
        num_sentence = input_batch.size(0)#句子数
        seql = input_batch.size(1)#最长词数
        input = self.wemb(input_batch)# [ num_sentence, seql, 32]

        _l = []

        for i in range(seql):
            data = torch.cat([input.narrow(1,i,1),last_state],dim=-1)# [num_sentence ,1 , 32+32]
            last_state = self.net(data)# [num_sentence ,1 , 32] -->隐状态
            _l.append(last_state)
        output = self.classifier(torch.cat(_l,dim=1))
        return output

    def decode(self,input,last_state):

        input = self.wemb(input)#【1，32】
        data = torch.cat([input,last_state],dim=1)
        last_state = self.net(data)
        out = torch.argmax(self.classifier(last_state),dim=1)
        return out,last_state


















