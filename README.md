## Forest Department Project (WIP)
This repository contains the `classification` model for the tasks required to be accomplised as a part of the Forest Department Project done by Aerodynamics Club, BITS Goa.

## Project Description
In this project we aim to 3D Map Mangrove Forests around Goa aerially. We use a DJI Phantom Pro v2. The collected images are then post processed and stiched together to obtain a 3D Map of the area. As a part of the project, we built a Machine Learning Model to classify the species observed in the forests. We built our dataset from freely available data online, augmented it an used the same for training.

## Classification model (WIP)
We performed Transfer Learning on pretrained `Resnet18` and `VGG` models from Pytorch. We use the ensemble of the two for inference.

| Model         | Epochs trained | Accuracy | F1 Score |
| :---          |       :---:    | :---:    | :---:    |
|`Resnet18`     | 30            |   81.67   | 0.81     |
|`VGG`          | 10             | 83.165   |   0.81   |
|`Ensemble`     | -              | 86.334   |   0.86   |

### Confusion Matrix
![Confusion Matrix](fig/heatmap.png) 

Confusion matrix generated using the ensemble model.

### Sample Results
The model was tested on real world data collected from our visits to Mangrove forests. More sample results can be found in the `fig` directory.
![Sample](fig/IMG_0629_test.png)
![Sample 2](fig/IMG_0630_test.png)


