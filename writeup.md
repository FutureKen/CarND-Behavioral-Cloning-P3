**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visual.jpeg "Loss Visualization"
[image2]: ./examples/cnn-architecture.png "Architecture"
[image3]: ./examples/c1.jpg "Recovery Image"
[image4]: ./examples/c2.jpg "Recovery Image"
[image5]: ./examples/c3.jpg "Recovery Image"
[image6]: ./examples/center_example.jpg "Normal Image"
[image7]: ./examples/fail.JPG "Failed section"
[image8]: ./examples/fail2.JPG "Failed section"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* run1.mp4 containing the video record of 1st track
* run2.mp4 containing the video record of 2nd track
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a powerful convolution neural network published by [nVidia](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), the model consists of a normalization from bottom, then followed by five convolutional layers, including three 5x5 filter sized layers and two 3x3 filter sized layers, followed by four fully connected layers.  (model.py lines 50-59) 

The model includes RELU layers to introduce nonlinearity (code line 50), and the data is normalized in the model using a Keras lambda layer (code line 46). Each image is cropped so that those trees, mountains and the vehicle head parts in the image won't be a part of the model, this also shortens the training time.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 62). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 60).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. In the final result I used the center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the LeNet, I tried to apply it in the beginning because it has simple architecture, so that I can train my data with it quickly and make some optimization later after I get everything running.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added a normalization layer and a cropping layer.

Then I switched to the CNN model published by nVidia team, and the result turned out to be better. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I decide to train more data at those sections.

At the end of the process, the vehicle is able to drive autonomously and smoothly around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 50-59) consisted of a convolution neural network with the following layers and layer sizes

Layer     | Size
-------- | ---
Lambda | (160, 320, 3)
Cropping2D    | (80, 320, 3)
Convolution2D_1     | (38, 158, 24)
Convolution2D_2     | (17, 77, 36)
Convolution2D_3     | (7, 37, 48)
Convolution2D_4     | (5, 35, 64)
Convolution2D_5     | (3, 33, 64)
Flatten     | (6336)
Dense_1		| (100)
Dense_2		| (50)
Dense_3		| (10)
Dense_4		| (1)

Here is a visualization of the architecture 

![alt text][image2]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image6]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to run safely within the road, there are some sections the vehicle rran out of the road or hit the barrier like the following picture shows.

![alt text][image7]
![alt text][image8]

 These images show what a recovery looks like starting from curb of road to center of road :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track at those sections twice in order to get more data points.

To augment the data sat, I also tried flipped images and angles thinking that this would reduce the MSE and making the result better. Unfortunately it only worsen the result so I have to discard this approach at last.

In general, I trained the following tracks with

First track
:   Two circle driving forward, one circle driving backward
:   Driving at curves once each, driving at failed places twice each

Second track
:	Three circles driving forward
:	Driving at failed places twice each

After the collection process, I had 53592 number of data points. I then preprocessed this data by using Lambda normalization and cropping.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by that it stopped reducing valicdation error after third epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.

The following picture shows the visualized loss.
![alt text][image1]