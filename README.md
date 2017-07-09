#### Behavioural Cloning.

The objective of this project is to train a model to enable driving of the car around test track in udacity simulator.  
This writeup gives an overview of the file structure, the design consideration that went into the project and results.

##### Folder Structure:
Main files part of this submission are:
- train.py: Implementation of models used for this project.
- imageHandler.py: Helper class to training.
- drive.py : Driving using the trained model.
- model,json: Model architecture config.
- model.h5 : Binary file with trained weights.

##### Running the model:
Model can be tested by executing:  
python drive.py model.h5

##### Model Architecture:
Two different models were looked into during the implementation phase, they were [comma.ai](https://arxiv.org/abs/1608.01230) model and [Nvidia Model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)   
As part of final implementation a modified version of nvidia model was used. The architecture of final trained model is shown below.
![Trained model](./model.png)
- In windows due to the [issue](https://stackoverflow.com/questions/41847376/keras-model-to-json-error-rawunicodeescape-codec-cant-decode-bytes-in-posi) with unicode parsing in lambda layer of tensorflow, instead of adding normalization layer, each image is first normalized as part of the batch generator in _imageHandler.py: process_batch()_.
- Batch normalization is used after Convolution layers and Dense layers.
- RELU Activation was used as activation layer.
- Max pooling was done after each convolutional layer.
- 'same' image padding was used instead of 'valid'.
- As an attempt to reduce over fitting dropout of 0.2 is added after convolution layers and a dropout of 0.5 is added after dense layers.
   -These changes were done based on driving accuracy on the simulator and was a trial and error approach.
- Though Adam optimizer was used for optimization, it was set to an initial training rate of 0.0001 instead of default.
- An 80: 20 split of randomly shuffle data was used to train and validate image.
- Initially training was done with an epoch of 30 with an EarlyStopping callback registered. Though validation error decreased till 30 epochs, the model was slow in reacting to corners in simulator and hence epoch was reduced to 20.

##### Data collection strategy.
- The quality of the collected data had significant impact on the driving accuracy in this project.
- An initial attempt was made to collect extensive data in straight and driving in opposite direction in the track-1. But the model failed on bridge corners.
  In the final approach following were the collected data:
- straight-recovery: 20313(including left, center and right camera) images of slow recovery from side.  
  ![straight recovery](./imgs/center_2017_04_21_17_35_49_845.jpg)
- wiggle : 23655 images of fast course correction between the lane lines.  
  ![wiggle](./imgs/center_2017_04_22_23_35_45_301.jpg)
- recover-unpatched : 2244 images of recovering near special cases of track, including curve after the bridge.  
  ![unpatched](imgs/center_2017_04_22_23_57_33_281.jpg)
- center : 20976 image of driving with vehicle on center of the track.  
  ![center](./imgs/center_2017_04_23_03_24_11_543.jpg)
- Data collection was made with speed in and around 20, also throttle was limited to keep the trained model within speed range of 20.

##### Reducing overfitting:
- Dropout was introduced to reduce overfitting for both convolution and dense layers.
- EarlyStopping callback was registered to stop training in case validation accuracy failing to improve.
- Images were randomly perturbed, images were sheared or flipped randomly to better generalize the model.
- Angle correction were added to the left and right camera images to help in steering recovery.
- 
##### Conclusion:
- A recorded [video](./video.mp4) of the model is part of this submission.
- The model was extremely susceptible to the type of data collected, as such a much more generalized model using transfer learning, from model trained on actual road scenes need to be tried.
- In future, VGG net needs to be tested to see how it fairs on the track.
