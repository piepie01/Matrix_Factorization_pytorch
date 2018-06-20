import numpy as np
import sys
import math
from sklearn.model_selection import train_test_split
def read_test(path):
    with open(path, 'r') as f:
        s = f.read().split('\n')[1:-1]
    X = []
    for i in s:
        x = []
        infor = list(map(int, i.split(',')))
        x.append([infor[1]])
        x.append([infor[2]])
        X.append(x)
    return X
class DataFiles():
    def __init__(self, train_file):
        #self.user_file = user_file
        #self.movie_file = movie_file
        self.train_file = train_file
    def read_data(self):
        print("Reading data...")
        #self.user = self._read_user()
        3self.movie = self._read_movie()
        self.train_X, self.train_Y = self._read_train()
        self.train_X, self.valid_X, self.train_Y, self.valid_Y = self._train_split()
        #self.past_rate = self._past_view()
    def _read_user(self):
        with open(self.user_file,'r') as f:
            s = f.read().split('\n')[1:-1]
        d = {}
        for item in s:
            people = item.split('::')
            gen = 0 if people[1] == 'M' else 1
            age = int(people[2])
            occ = int(people[3])
            d[int(people[0])] = [gen, age, occ]
        return d
    def _read_movie(self):
        with open(self.movie_file,'r',encoding = 'ISO-8859-1') as f:
            s = f.read().split('\n')[1:-1]
        ind = [int(i.split('::')[0]) for i in s]
        name = [i.split('::')[1] for i in s]
        feature = [i.split('::')[2].split('|') for i in s]
        feature2ind = []
        for i in feature:
            for item in i:
                if item not in feature2ind:
                    feature2ind.append(item)
        new = {}
        self.movie_kinds = len(feature2ind)
        for i,txt in enumerate(feature):
            tmp = [feature2ind.index(j) for j in txt]
            q = [0 for _ in range(len(feature2ind))]
            for p,j in enumerate(tmp):
                q[j] = 1
            new[ind[i]] = q
        return new
    def _read_train(self):
        self.max_user = 0
        self.max_movie = 0
        with open(self.train_file, 'r') as f:
            s = f.read().split('\n')[1:-1]
        X = []
        Y = []
        for i in s:
            x = []
            infor = list(map(int, i.split(',')))
            if infor[1] > self.max_user:
                self.max_user = infor[1]
            if infor[2] > self.max_movie:
                self.max_movie = infor[2]
            x.append([infor[1]])
            x.append([infor[2]])
            X.append(x)
            Y.append(infor[3])
        return X,Y
    def _past_view(self):
        past = [0 for _ in range(len(self.user) + 1)]
        num = [0 for _ in range(len(self.user)+1)]
        for x,y in zip(self.train_X, self.train_Y):
            past[x[0][0]] += y
            num[x[0][0]] += 1
        for i in range(len(past)):
            if past[i] != 0:
                #past[i] = past[i] / (math.sqrt(num[i]) * 10)

                past[i] = past[i] / num[i]
        return past
    def _train_split(self, rate = 0.02):
        train_X,valid_X,train_Y,valid_Y = train_test_split(self.train_X,self.train_Y,test_size = rate,random_state = 127)
        return train_X, valid_X, train_Y, valid_Y
    def view_data(self):
        print("----data information----")
        #print("Movie class = {}".format(self.movie_kinds))
        print("User kinds = {}".format(self.max_user))
        print("Movie kinds = {}".format(self.max_movie))
        print("----train_X({}), train_Y({})----".format(len(self.train_X), len(self.train_Y)))
        print("user_id, movie_id")
        for i in range(3):
            print(self.train_X[i], self.train_Y[i], sep = ', ')
        print("----valid_X({}), valid_Y({})----".format(len(self.valid_X), len(self.valid_Y)))
        print("user_id, movie_id")
        for i in range(3):
            print(self.valid_X[i], self.valid_Y[i], sep = ', ')
        #print("----user----")
        #print("id, [gender, age, occupation]")
        #for i in list(self.user.keys())[:2]:
        #    print(i, self.user[i], sep = ', ')
        #print('...')
        #print(list(self.user.keys())[-1], self.user[ list(self.user.keys())[-1] ])
        #print("----movie----")
        #for i in list(self.movie.keys())[:2]:
        #    print(i, self.movie[i], sep = ', ')
        #print('...')
        #print(list(self.movie.keys())[-1], self.movie[ list(self.movie.keys())[-1] ])
