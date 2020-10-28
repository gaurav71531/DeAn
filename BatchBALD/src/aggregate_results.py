import pickle as pk
import os
import numpy as np

def aggregate_results(ds, str_):
    resultsDir = os.path.join('../Results', ds)
    Experiments_All_Accuracy = [1]
    e = 0
    for f in os.listdir(resultsDir):
        if f.split('.')[0][:len(str_)] == str_:
            all_acc = pk.load(open(os.path.join(resultsDir, f), 'rb'))
            Experiments_All_Accuracy[e] = all_acc
            Experiments_All_Accuracy += [1]
            e += 1
    Experiments_All_Accuracy = Experiments_All_Accuracy[:-1]
    outData = {'Experiments_All_Accuracy': Experiments_All_Accuracy}
    fileName = str_ + '.p'
    pk.dump(outData, open(os.path.join(resultsDir, fileName), 'wb'))


if __name__ == '__main__':
    aggregate_results('MNIST', 'AL_BALD_b_200_Exp_1')
