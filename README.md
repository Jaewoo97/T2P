# T2P
Official repository of: Multi-agent Long-term 3D Human Pose Forecasting via Interaction-aware Trajectory Conditioning (CVPR 2024 Highlight)
[[paper](https://openaccess.thecvf.com/content/CVPR2024/html/Jeong_Multi-agent_Long-term_3D_Human_Pose_Forecasting_via_Interaction-aware_Trajectory_Conditioning_CVPR_2024_paper.html)] [[project page](https://jaewoo97.github.io/t2p_/)]

## JRDB-GMP Dataset
We upload an updated version of the JRDB dataset parser (3D joints => SMPL parameters for pose). 

1. Set conda environment based on [BEV](https://github.com/Arthur151/ROMP). Also install torch-geometric. (Tested version: Python 3.11.9, cuda 12.1, torch 2.3.0, torch-geometric 2.5.3)
2. Download the original dataset from [JRDB](https://jrdb.erc.monash.edu/). Change the default_save_dir in `preprocess_1st_jrdb.py` accordingly.
3. Download the preprocessed robot odometry files (.npy) from releases. Change the directory of odometry_base in `preprocess_1st_jrdb.py` accordingly. Thanks to [human-scene-transformer](https://github.com/google-research/human-scene-transformer) for sharing your work.
4. Process the 1st and 2nd in order (`preprocess_1st_jrdb.py`, `preprocess_2nd_jrdb.py`).
```
python preprocess_1st_jrdb.py
```
`preprocess_1st_jrdb.py`: Processes the trajectory information from 3D bounding boxes, and 3D pose is extracted by [BEV](https://github.com/Arthur151/ROMP). Theta parameters of SMPL (24X3) is used as pose information. Each frame is preprocessed independently.
```
python preprocess_2nd_jrdb.py
```
`preprocess_2nd_jrdb.py`: Parses each scene into .pt file. The data is saved as TemporalData class, a format used by [HiVT](https://github.com/ZikangZhou/HiVT). The parameters are set to parse the data in 2.5FPS.

## Training
1. Preprocessing JRDB
JRDB: Adjust the directories to preprocessed files in `dataset/t2p_dataset.py` accordingly after preprocessing the desired dataset from above.

2. Preprocessing CMU-Mocap / 3DPW: These datasets require a separate preprocessing step. Download raw dataset files from release and run `dataset/3dpw_hivt.py`, `dataset/Mocap_UMPM_hivt.py` with process_dir adjusted to your environment.
```
python dataset/3dpw_hivt.py
python dataset/Mocap_UMPM_hivt.py
```
3. Training
Setup environment by running `install_dependencies.sh` with python 3.8 and cuda 12.1. Run `lightning_train.py`.

## Acknowledgements
- Thanks to [human-scene-transformer](https://github.com/google-research/human-scene-transformer), [JRDB](https://jrdb.erc.monash.edu/), [BEV](https://github.com/Arthur151/ROMP), [HiVT](https://github.com/ZikangZhou/HiVT) for sharing your work.
