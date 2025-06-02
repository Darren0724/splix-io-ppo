

# The splix.io AI based on deep reinforcement learning (Phase One - Territory Occupying AI)

This repository is the official implementation of NTU-DRL Class Final Project.

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

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

We have a nice result for training an AI playing splix.io in the phase 1, 士氣大振.

We can try to do phase 2 (multi-agent) in the future.
