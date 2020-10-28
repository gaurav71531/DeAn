import numpy as np
import os

if __name__ == '__main__':
    
    methodList = ['Random-noise', 'BALD-noise', 'k-center-greedy-noise',
                 'maxEnt-noise', 'VAAL-noise', 'cluster-noise']
    exp_no = [
        [35, 36, 37, 38, 39, 40],
        [41, 42, 43, 44, 45, 46],
        [47, 48, 49, 50, 51, 52],
        [53, 54, 55, 56, 57, 58],
    ]
    expList = [20] * len(methodList)

    dataType = 'MNIST'
    trainNoiseLayer = True
    Queries=100
    nb_epoch = 50

    for oii, onp in enumerate([0.1, 0.2, 0.3, 0.4]):

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