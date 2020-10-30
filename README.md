# Noisy Batch Active Learning with Deterministic Annealing

Gaurav Gupta, Anit Kumar Sahu, and Wan-Yi Lin

![Image](http://scf.usc.edu/~ggaurav/pics/DeAn.png)

**Language:** The code package is available in Python using Pytorch libary

The current code also uses the code from following git repositories for comparison:

1. https://github.com/sinhasam/vaal
2. https://github.com/BlackHC/BatchBALD

For running the comparisons, the included versions of the above code repositories are modified to have noisy oracle and the denoising layer

The experimemts were done using cuda 10.1. For running the simualtions, make sure to install the Pytorch using
```
conda install pytorch==1.2.0 torchvision==0.4.0 -c pytorch
```

and the following
```
pip install -r requirements.txt
```
To reproduce results, run files with datasets name, for example
```
python run_mnist.py
```

For EMNIST, if the http link error occur then replace the Pytorch current url for emnist with the following:
```
https://cloudstor.aarnet.edu.au/plus/s/ZNmuFiuQTqZlu9W/download
```

The figures are generated on Linux using Matplotlib, for using the current code make sure to install the following latex components

```
sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended 
```

## Citation
For using the code in your work, please cite
```
@misc{gupta2020noisy,
      title={Noisy Batch Active Learning with Deterministic Annealing}, 
      author={Gaurav Gupta and Anit Kumar Sahu and Wan-Yi Lin},
      year={2020},
      eprint={1909.12473},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```