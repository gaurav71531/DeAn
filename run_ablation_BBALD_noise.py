import numpy as np
import os
from BatchBALD.src.aggregate_results import aggregate_results


if __name__ == '__main__':
    onp = [0.1, 0.2, 0.3, 0.4]
    exp_id = [1, 2, 3, 4]
    num_experiments = 5
    ds = 'MNIST'
    sim_file_name = 'run_experiment.py'
    experiment_description = 'al_DeAn'
    batch_size = 128
    epochs = 50
    early_stopping_patience = 100 # put greater than epochs for no-stopping
    num_inference_samples = 100 # dropout iterations
    available_sample_k = 100
    num_al_iterations = 5
    initial_samples_per_class = 4
    target_num_acquired_samples = available_sample_k*5 + initial_samples_per_class*10 + 1

    for ii, oracle_p in enumerate( onp):
        for e in range(num_experiments):
            experiment_task_id = 'AL_BBALD_b_' + np.str(available_sample_k) + '_Exp_' + np.str(exp_id[ii]) + '_' + np.str(e)

            strUse =  ('python -W ignore::UserWarning ' + 'BatchBALD/src/' + sim_file_name 
                                + ' --ds ' + ds
                                + ' --batch_size ' + np.str(batch_size)
                                + ' --epochs ' + np.str(epochs)
                                + ' --num_inference_samples ' + np.str(num_inference_samples)
                                + ' --available_sample_k ' + np.str(available_sample_k)
                                + ' --target_num_acquired_samples ' + np.str(target_num_acquired_samples)
                                + ' --num_al_iterations ' + np.str(num_al_iterations)
                                + ' --initial_samples_per_class ' + np.str(initial_samples_per_class)
                                + ' --early_stopping_patience ' + np.str(early_stopping_patience)
                                + ' --experiment_description ' + experiment_description
                                + ' --experiment_task_id ' + experiment_task_id
                                + ' --onp ' + np.str(oracle_p)
                        )
        
            os.system('echo ' + "\"" +  strUse + "\"")
            os.system(strUse)
        
        experiment_task_id = 'AL_BBALD_b_' + np.str(available_sample_k) + '_Exp_' + np.str(exp_id[ii])
        aggregate_results(ds, experiment_task_id)
