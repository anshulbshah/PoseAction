# Pose and Joint-Aware Action Recognition

Code and Pre-processed data for the paper Pose and Joint-Aware Action Recognition accepted to WACV 2022

[[`Paper`](https://openaccess.thecvf.com/content/WACV2022/papers/Shah_Pose_and_Joint-Aware_Action_Recognition_WACV_2022_paper.pdf)] [[`Video`](https://youtu.be/BqaOlF_LOMA)]

## Set-up environment
- Tested with Python Version : 3.7.11
  
Follow one of the following to set up the environment:
- A) Install from conda environment : `conda env create -f environment.yml`
- B) The code mainly requires the following packages : torch, torchvision, puytorch 
  - Install one package at a time :
  - `conda create -n pose_action python=3.7`
  - `conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge`
  - `pip install opencv-python matplotlib wandb tqdm joblib scipy scikit-learn`
- C) Make an account on wandb and make required changes to `train.py` L36


## Prepare data
- `mkdir data`
- `mkdir metadata`
-  Download data from [here](http://www.cis.jhu.edu/~ashah/PoseAction/data/). Extract the tar files with folder structure `data/$dataset/openpose_COCO_3/`
-  Download metadata from [here](http://www.cis.jhu.edu/~ashah/PoseAction/metadata.tar.gz). Extract the tar files to `data/metadata`

## Training scripts
- Example : `bash sample_scripts/hmdb.sh`

## Raw heatmaps
We also provide raw heatmaps [here](https://1drv.ms/u/s!AlAjgCeVY_IrgY40FMWKAsiO5-Opmw?e=N8e4A6). OpenPose was used to extract these. Please take a look at function `final_extract_hmdb` in `utils.py` for an example function to extract pose data. 

## Citation
If you find this repository useful in your work, please cite us! 
```
@InProceedings{Shah_2022_WACV,
    author    = {Shah, Anshul and Mishra, Shlok and Bansal, Ankan and Chen, Jun-Cheng and Chellappa, Rama and Shrivastava, Abhinav},
    title     = {Pose and Joint-Aware Action Recognition},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2022},
    pages     = {3850-3860}
}
```

