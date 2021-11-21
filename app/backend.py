"""File that can decorate the output and show you clean plots"""
import numpy as np
from classifier import CLASSES
from classifier import Classifier
import matplotlib.pyplot as plt
import sys


classifier = Classifier()

def decorate_results(image, fi=None, f=None):

    preds_ens, preds_resnet, preds_vgg, X_ens, X_resnet, X_vgg = classifier.predict_image(image, f=f, verbose=True)

    # make the figure
    fig = plt.figure(figsize=(16, 8), constrained_layout=True)

    gs = fig.add_gridspec(3, 3)

    # image occupies 0 to 2
    img_axis = fig.add_subplot(gs[0:-1])
    img_axis.set_title(f"{CLASSES[preds_ens]}")
    img_axis.imshow(plt.imread(image), interpolation="nearest")

    # Resnet
    percentages = [i.item() * 100 for i in X_resnet[0]]
    resnet_axis = fig.add_subplot(gs[0, -1])
    resnet_axis.set_title("Resnet")
    mlabels = [CLASSES[i] if percentages[i] > np.median(percentages) + np.mean(percentages) * 0.2 else "" for i in range(len(CLASSES))]
    explode = np.zeros_like(percentages)
    explode[np.argmax(percentages)] = 0.25
    resnet_axis.pie(percentages, labels=mlabels, explode=explode, shadow=True)

    # VGG
    percentages = [i.item() * 100 for i in X_vgg[0]]
    vgg_axis = fig.add_subplot(gs[1, -1])
    vgg_axis.set_title("VGG")
    mlabels = [CLASSES[i] if percentages[i] > np.median(percentages) + np.mean(percentages) * 0.2 else "" for i in range(len(CLASSES))]
    explode = np.zeros_like(percentages)
    explode[np.argmax(percentages)] = 0.25
    vgg_axis.pie(percentages, labels=mlabels, explode=explode, shadow=True)

    # Ensemble
    percentages = [i.item() * 100 for i in X_ens[0]]
    ens_axis = fig.add_subplot(gs[2, -1])
    ens_axis.set_title("Ensemble")
    mlabels = [CLASSES[i] if percentages[i] > np.median(percentages) + np.mean(percentages) * 0.2 else "" for i in range(len(CLASSES))]
    explode = np.zeros_like(percentages)
    explode[np.argmax(percentages)] = 0.25
    ens_axis.pie(percentages, labels=mlabels, explode=explode, shadow=True)

    # Save the image in the path
    if fi is not None:
        plt.savefig(fi, bbox_inches="tight")

    
