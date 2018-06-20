import torch.nn.functional as functional
import torch
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
class clr_conv(Dataset):
    def __init__(self,train_X,train_Y):
        self.Y = train_Y
        self.X = train_X
        self.length = len(train_X)
    def __len__(self):
        return self.length
    def __getitem__(self,index):
        y = [self.Y[index]]
        x0 = torch.LongTensor(self.X[index][0])
        x1 = torch.LongTensor(self.X[index][1])
        y = torch.FloatTensor(y)
        return x0,x1,y
class test_clr_conv(Dataset):
    def __init__(self,train_X):
        self.X = train_X
        self.length = len(train_X)
    def __len__(self):
        return self.length
    def __getitem__(self,index):
        x0 = torch.LongTensor(self.X[index][0])
        x1 = torch.LongTensor(self.X[index][1])
        return x0,x1
class Net(nn.Module):
    def __init__(self, user_len, movie_len, embedding_size):
        super(Net, self).__init__()
        self.embed_0 = nn.Embedding(user_len+1, embedding_size, sparse = True)
        self.embed_1 = nn.Embedding(movie_len+1, embedding_size, sparse = True)
        self.embed_0_0 = nn.Embedding(user_len+1, 1, sparse = True)
        self.embed_1_0 = nn.Embedding(movie_len+1, 1, sparse = True)
        self.embedding_size = embedding_size
    def forward(self, seq0, seq1):
        out0 = self.embed_0(seq0)
        out1 = self.embed_1(seq1)
        out0 = out0.view(out0.size()[0],self.embedding_size)
        out1 = out1.view(out1.size()[0],self.embedding_size)
        out0 = functional.rrelu(out0)# + p
        out1 = functional.rrelu(out1)

        bias_0 = self.embed_0_0(seq0)
        bias_1 = self.embed_1_0(seq1)
        out = bias_0 + bias_1 #+ p.view(mean.size()[0],1,1)
        out = out.view(out.size()[0],1)
        ma = (out0 * out1).sum(1)
        ma = ma.view(ma.size()[0],1)
        return ma+out
def weights_init(net):
    classname = net.__class__.__name__
    if classname.find('Embedding') != -1:
        nn.init.xavier_normal(net.weight.data)
class Matrix_Factorization():
    def __init__(self, user_len, movie_len, embedding_size, learning_rate, cuda, pre_train, model):
        self.cuda = cuda
        if cuda:
            self.net = Net(user_len, movie_len, embedding_size).cuda()
        else:
            self.net = Net(user_len, movie_len, embedding_size)
        if pre_train == True:
            self.net.load_state_dict(model)
        else:
            self.net.apply(weights_init)
            if cuda:
                self.criterion = nn.MSELoss().cuda()
            else:
                self.criterion = nn.MSELoss()
            self.opt = torch.optim.SparseAdam(self.net.parameters(), lr = learning_rate)#,weight_decay=1e-5)
        #opt = torch.optim.SGD(net.parameters(), lr = 1e-3,weight_decay=1e-5,  momentum=0.9)
    def fit(self, Data, epochs, batch_size, verbose_step, verbose_test, save_file):

        self.min_rmse = 10.0
        conv = clr_conv(Data.train_X,Data.train_Y)
        data_loader = DataLoader(conv, shuffle = True, batch_size = 128)
        vali_conv = clr_conv(Data.valid_X,Data.valid_Y)
        vali_data_loader = DataLoader(vali_conv, shuffle = False, batch_size = 128)

        for epoch in range(epochs):
            cnt = 1
            for user,movie,y in data_loader:
                self.net.train()
                self.opt.zero_grad()
                if self.cuda:
                    user = Variable(user.cuda())
                    movie = Variable(movie.cuda())
                    oup = Variable(y.cuda())
                else:
                    user = Variable(user)
                    movie = Variable(movie)
                    oup = Variable(y)
                out = self.net(user,movie)
                loss = self.criterion(out, oup)
                if cnt % verbose_step == 0:
                    print('[epoch {}] [stet {}] loss : {}'.format(epoch, cnt, loss.data[0]))
                loss.backward()
                self.opt.step()
                if cnt % verbose_test == 0:
                    score = self._fit_valid(vali_data_loader, Data.valid_Y)
                    if score < self.min_rmse:
                        torch.save({'model' : self.net.state_dict(), 'user_kinds' : Data.max_user, 'movie_kinds' : Data.max_movie},save_file)
                        self.min_rmse = score
                cnt+=1
    def _fit_valid(self, loader, valid_Y):
        self.net.eval()
        pre = []
        total = 0.0
        cnt = 0
        for x0,x1,y in loader:
            if self.cuda:
                inp0 = Variable(x0.cuda())
                inp1 = Variable(x1.cuda())
                oup = Variable(y.cuda())
            else:
                inp0 = Variable(x0)
                inp1 = Variable(x1)
                oup = Variable(y)
            out = self.net(inp0,inp1)
            total += self.criterion(out, oup)
            cnt += 1
            out1 = out.data.cpu().numpy()
            for item in out1:
                pre.append(item[0])
            #del out
        pre = np.array(pre)
        pre = pre.clip(1.0, 5.0)
        rmse = np.sqrt(((pre-valid_Y)**2).mean())
        print('----------------{}-------------loss : {}'.format(rmse, total.data[0]/cnt))
        self.net.train()
        return rmse
    def predict(self, test_X):
        self.net.eval()
        test_conv = test_clr_conv(test_X)
        test_data_loader = DataLoader(test_conv, shuffle = False, batch_size = 128)
        pre = []
        for x0,x1 in test_data_loader:
            if self.cuda:
                inp0 = Variable(x0.cuda())
                inp1 = Variable(x1.cuda())
            else:
                inp0 = Variable(x0)
                inp1 = Variable(x1)
            out = self.net(inp0,inp1)
            out = out.data.cpu().numpy()
            for item in out:
                pre.append(item[0])
        pre = np.array(pre)
        pre = pre.clip(1.0, 5.0)
        return pre
