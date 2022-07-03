from math import sqrt

import h5py
import torch

from batch_RNN_model import *


f = h5py.File("final_result1.hdf5", "r")
# vocab_size = f['nword'][:].item()
vocab_size = f['nword'][()]

# GPU
device = torch.device("cuda:0")
'''
device = torch.devixe("cpu")表示使用cpu
device = torch.device("cuda")表示使用GPU
'''

# 固定随机种子
torch.manual_seed(0)    # GPU: torch.cuda.manual_seed(n)

Model = TestModel(vocab_size)#模型初始化，init传入词典大小
Model.to(device)

# 参数初始化
for param in Model.parameters():    # weight, bias
    # nn.init.xavier_uniform_(param, gain=1)    # param.dim() > 1
    rang = sqrt(1 / param.size(-1))
    nn.init.uniform_(param, a=-rang, b=rang)

# loss
loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')      # ignore pad, loss = sum()
loss_fn.to(device)

# Optimizer
optm = torch.optim.Adam(Model.parameters())
# optm.to(device)

# train
curb = 0
context_size = 3
epoch = 20
for i in range(epoch):
    for index in f["group"]:
        input = torch.LongTensor(f["group"][index][:])
        ndata = input.size(1)#词数
        num_sen = input.size(0)#句子数
        # target = input.narrow(1, context_size, ndata)
        target = input
        #句子数 * n-3

        input = input.to(device)
        target = target.to(device)

        dim_0 = torch.nn.Parameter(torch.zeros(num_sen,1, 32))
        dim_0 = dim_0.to(device)        # print('output 的大小是', output.size())--> output 的大小是 torch.Size([1, 32])
        # forward
        output = Model(input,dim_0)
        output = output.transpose(1, 2)
        output = output.to(device)
        #句子数  *  n-3  *  词数  --> 句子数 * 词数 * n-3
        loss = loss_fn(output, target)

        # backward
        loss.backward()

        # 更新参数
        optm.step()
        optm.zero_grad()  # 梯度清零

        # save
        curb += 1
        print(curb)
        if curb % 100 == 0:
            print("训练次数：{}, loss = {}".format(curb, loss.item()/vocab_size))
            torch.save(Model, "./true_result/rnn_{}.pth".format(curb))

f.close()


# RuntimeError: Expected target size [10, 8838], got [10, 253]






