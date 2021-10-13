## Forest Department Project (WIP)
This repository contains the `classification` model for the tasks required to be accomplised as a part of the Forst Department Project done by Aerodynamics Club, BITS Goa.

### Classification model (WIP)
We performed Transfer Learning on pretrained `Resnet18` and `VGG` models from Pytorch. We use the ensemble of the two for inference. \

| Model         | Epochs trained | Accuracy | F1 Score |
| :---          |       :---:    | :---:    | :---:    |
|`Resnet18`     | 30            |   81.67   | 0.81     |
|`VGG`          | 10             | 83.165   |   0.81   |
|`Ensemble`     | -              | 86.334   |   0.86   |

## Heatmap
<img src="SuhrudhSarathy/ForestDeptProj/fig/heatmap.png" alt="Heatmap after Ensemble" style="height: 100px; width:100px;"/>


