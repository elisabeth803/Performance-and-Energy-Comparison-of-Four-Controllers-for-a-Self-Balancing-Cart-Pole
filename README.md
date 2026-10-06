# TEK4090 inverted cart pole control simulation

The structure of the project directory: 
```text
MCT4053_exam_2026
├── snutter/                    # segmented versions of the videos
├── videoer/                    # raw, unprocessed videoes
├── cnn_simple_norm_params.npz  # parameters such as mean and standard deviation for the model
├── cnn_simple.h5               # ready-to-use model-file
├── csv_to_dataset.py           
├── landmark_to_csv.py
├── landmarks_all_videos.csv
├── mocap_dataset.npz
├── pose_landmarker_lite.task
├── README.md
├── realtime_classification.py
├── requirements.txt
├── segment_videos.py
├── test.py
├── train_simple_cnn.py
└── trim_videos.py
``` 

### SETUP
1) First we create environment and install the dependencies
```console
conda env create -f environment.yml
```
2) Activate envirmonamt
```console
conda activate cartPole-env
```
3) Test to see if everything is correct.
```console
python -c "import numpy, scipy, matplotlib, numba, control, skfuzzy, pybullet; print('ok')"

```

4) Run the code
```console
python .\pendulumSim.py --controller lqr --T 10
```

5) Test more functionality
```console
python .\pendulumSim.py --help
```
