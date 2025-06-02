# The splix.io AI based on deep reinforcement learning (Phase One - Territory Occupying AI)

This repository is the official implementation of NTU-DRL Class Final Project.

## Introduction

Train an AI to play splix.io by DRL method.

## Score

|         |   reward            | 
| ------------------ |---------------- | 
| each cell enclosed   |   +1.0     | 
| invalid move | -1.0 | 
|  time step | -0.1 | 



## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

"It's recommended to use venv to run it."



## Training

To train the model(s) in the paper, cd to the folder and run this command:

```train
python <training_file_name>.py 
```

You can modify the parameters like episodes, load_model and save_video path for usage.

## Evaluation

To just evaluation the model, turn the flag to True and run this command. 

```eval
python <training_file_name>.py 
```



## Pre-trained Models

There's no pre-trained model for our project.



## Results

Our model achieves the following performance by our testing:

| Model name         | avg. reward  | std |
| ------------------ |---------------- | -------------- |
| Original env   |   -0.6     |  0.02           |
| Reward Shaping | 83.4 | 46.6 |
| Eight Directional Sign Arrays and Last Action | 110.93 | 125.9 |
| Distance Arrays and Last Action | 150.18 |  126.92 |



## Contributing

We have a nice result for training an AI playing splix.io in the phase 1, 士氣大振.

We can try to do phase 2 (multi-agent) in the future.
