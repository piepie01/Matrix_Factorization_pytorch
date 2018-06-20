from dataset import read_test
from model import Matrix_Factorization
import torch
import sys
def main(argv):
    test_X = read_test(argv[1])
    model_list = ['model0.th']
    ans = 0.0
    for name in model_list:
        model = torch.load('save/'+name)
        OuO = Matrix_Factorization(user_len = model['user_kinds'],
                                   movie_len = model['movie_kinds'],
                                   embedding_size = 500,
                                   learning_rate = 1e-3,
                                   pre_train = True,
                                   model = model['model'])
        ans += OuO.predict(test_X)
        #print(ans)
    with open(argv[2],'w') as f:
        print('TestDataID,Rating',file = f)
        for i,txt in enumerate(ans):
            print(i+1,txt/len(model_list), sep = ',',file = f)
if __name__ == "__main__":
    main(sys.argv)
