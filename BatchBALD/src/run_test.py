import numpy as np
import os


if __name__ == '__main__':
    num_experiments = 5
    sim_file_name = 'run_experiment.py'
    experiment_description = 'al_DeAn'
    batch_size = 128
    epochs = 50
    early_stopping_patience = 100 # put greater than epochs for no-stopping
    num_inference_samples = 100 # dropout iterations
    available_sample_k = 200
    target_num_acquired_samples = 1200
    initial_samples_per_class = 4
    # acquisition_method = 'AcquisitionMethod.independent'

    for e in range(num_experiments):
        experiment_task_id = 'test_' + np.str(e)

        strUse =  ('CUDA_VISIBLE_DEVICES=1 python -W ignore::UserWarning ' + sim_file_name 
                            + ' --batch_size ' + np.str(batch_size)
                            + ' --epochs ' + np.str(epochs)
                            + ' --num_inference_samples ' + np.str(num_inference_samples)
                            + ' --available_sample_k ' + np.str(available_sample_k)
                            + ' --target_num_acquired_samples ' + np.str(target_num_acquired_samples)
                            + ' --initial_samples_per_class ' + np.str(initial_samples_per_class)
                            # + ' --acquisition_method ' + acquisition_method
                            + ' --early_stopping_patience ' + np.str(early_stopping_patience)
                            + ' --experiment_description ' + experiment_description
                            + ' --experiment_task_id ' + experiment_task_id
                    )
    
        os.system('echo ' + "\"" +  strUse + "\"")
        os.system(strUse)