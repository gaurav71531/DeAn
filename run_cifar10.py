import numpy as np
import os

if __name__ == '__main__':
    
    methodList = ['Random', 'BALD', 'k-center-greedy', 'maxEnt', 'VAAL', 'cluster', 'cluster-noise']
    exp_no = [
        [1, 2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19, 20, 21],
        [21, 22, 23, 24, 25, 26, 27],
        [28, 29, 30, 31, 32, 33, 34],
    ]
    expList = [20] * len(methodList)

    dataType = 'CIFAR10'
    trainNoiseLayer = True
    Queries=5000
    nb_epoch = 50

    for oii, onp in enumerate([0, 0.1, 0.2, 0.3, 0.4]):

        for method, Experiments, exp_no_start in zip(methodList, expList, exp_no[oii]):

            if onp == 0 and method.split('-')[-1] == 'noise':
                continue

            if method == 'VAAL' or method == 'VAAL-noise':
                sim_file_name = 'al_run_sim_all1.py'
            else:
                sim_file_name = 'al_run_sim_all.py'

            strUse =  ('python ' + sim_file_name 
                    + ' -data ' + dataType
                    + ' -tn '
                    + ' -m ' + method
                    + ' -q ' + np.str(Queries)
                    + ' -ne ' + np.str(Experiments)
                    + ' -np ' + np.str(onp)
                    + ' -exp-no ' + np.str(exp_no_start) 
                    + ' -ep ' + np.str(nb_epoch)
            )
            os.system('echo ' + "\"" +  strUse + "\"")
            os.system(strUse)