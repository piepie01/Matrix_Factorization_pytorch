# Matrix_Factorization_pytorch

## Reading data

* After reading the training data, the data format is like ...
```bash
Reading data...
----data information----
User kinds = 6040
Movie kinds = 3952
----train_X(881875), train_Y(881875)----
user_id, movie_id
[[2586], [3481]], 4
[[3844], [1343]], 5
[[1358], [2506]], 5
----valid_X(17998), valid_Y(17998)----
user_id, movie_id
[[4065], [764]], 3
[[3614], [915]], 5
[[3174], [2375]], 2
```


* The sample training code
```python
from dataset import DataFiles
from model import Matrix_Factorization

if __name__ == "__main__":
    Data = DataFiles(train_file = "train.csv.xls")
    Data.read_data()
    Data.view_data()
    
    OuO = Matrix_Factorization(user_len = Data.max_user, # user kinds
                               movie_len = Data.max_movie, # movie kinds
                               embedding_size = 500, 
                               learning_rate = 1e-3,
                               cuda = False,
                               pre_train = False, # if there is a pre-trained model or model for prediction, turn it to True
                               model = None) # if pre_train == True, mention the model's path
    OuO.fit(Data = Data,
            epochs = 10,
            batch_size = 128,
            verbose_step = 100, # print on screen every x steps
            verbose_test = 1000, # check the validation data and save the model every x steps
            save_file = 'save/model.th')
```


* Some part of the data I used.
```bash
TrainDataID,UserID,MovieID,Rating
1,796,1193,5
2,796,661,3
3,796,914,3
4,796,3408,4
5,796,2355,5
6,796,1197,3
```
