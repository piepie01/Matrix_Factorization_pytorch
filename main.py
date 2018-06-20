#coding=utf8
from dataset import DataFiles
from model import Matrix_Factorization
import sys

if __name__ == "__main__":
    Data = DataFiles(train_file = "train.csv.xls")
    Data.read_data()
    Data.view_data()
    
    OuO = Matrix_Factorization(user_len = Data.max_user,
                               movie_len = Data.max_movie,
                               embedding_size = 500,
                               learning_rate = 1e-3,
                               pre_train = False,
                               model = None)
    OuO.fit(Data = Data,
            epochs = 10,
            batch_size = 128,
            verbose_step = 100,
            verbose_test = 1000,
            save_file = 'save/model.th')
