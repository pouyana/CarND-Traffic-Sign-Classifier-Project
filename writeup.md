# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/dist.png "Visualization"
[image2]: ./images/normalized.png "Normalized"
[image3]: ./images/image_from_interner.png "Traffic Signs"
[image4]: ./images/softmax_1.png "Softmax 1"
[image5]: ./images/softmax_2.png "Softmax 2"
[image6]: ./images/softmax_3.png "Softmax 3"
[image7]: ./images/softmax_4.png "Softmax 4"
[image8]: ./images/softmax_5.png "Softmax 5"
[image9]: ./images/softmax_6.png "Softmax 6"
[image10]: ./images/activation.png "Softmax 6"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is `34799`
* The size of the validation set is `4410`
* The size of test set is `12630`
* The shape of a traffic sign image is `32x32x3`
* The number of unique classes/labels in the data set is `43`

#### 2. Include an exploratory visualization of the dataset.

To visualize the database I tried the show the distributions of labels in different sets.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I tried to work with the grey scale images but, it did not help much
so I scarp the idea. The only action I did on the images was to normalizing them 
from -0.5 to .05 using the equalizeHist. For printing reason, as negative values dont
make sense, I added 0.5 to the images. But the network, trains with images with value from -0.5 to .5
for every color channel.

![alt text][image2]

I uses the image augmentation (https://github.com/aleju/imgaug)library. With
this it is possible to widen the training sets and have most of the image variants for the given training sets. The filter I used with this libray consist of:

- Zooming in/out of image with 20%
- Translate the image with 20% across each axis
- Rotate the image with 10 degrees across each axis
- Shear the image by 10 degrees across each axis

This operations were done randomly on the images, in random order and tried them 10 times.
So my training set is at the end 10 times bigger and contains more samples for the same images. With this, without new picture, I have more images to train the network with.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

For the sigma and mu I used the values: `0.1` and `0`

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x12 	|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x24 	|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 14x14x48    |
| Convolution 5x5		| 1x1 stride, valid padding, outputs 10x10x96   |
| RELU  				|           									|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x96 	|
| Flatten				| inputs 5x5x96, outputs 2400	                |
| Fully Connected		| inputs 2400, outputs 480          			|
| RELU  				|           									|
| drop Out  			| prob: 0.5           							|
| Fully Connected		| inputs 480, outputs 84            			|
| RELU  				|           									|
| drop Out  			| prob: 0.5           							|
| Softmax		        | inputs 84, outputs 43             			|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained the model with the AdamOptimizer, this is the simplest and most efficient optimizer available. The batch size I used was `1024` and the epochs were `16`. For the learning rate I choose a number small as much as possible `0.0009` which is very near to the `0.001` which was used in the course. For the drop outs probability, which explicitly are used in training I used the `0.5`. The layer depth and convulsion kernel size are also the parameters that are used.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of `0.9978`
* validation set accuracy of `0.9912`
* test set accuracy of `0.9778`

I started with the LENET that was introduced in the course, It was not accurate enough for the validation set `0.88` and for the training set it has `0.93`. I looked like an overfitting problem, so I tried to add the dropouts to the fully connected layer. The best solution to overfitting is to use dropout. The 0.5 probability was the best choice. I also tried to add more convulsion layers and max pools to the architecture. This caused the performance to be much higher.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image3]

The first image for no-passing has still some features in the background, It could be a little hard to classify. The road sign with 50Km/h max speed is a little bit rotated and has sun reflection, this could also cause problems, for the
others I would say I could work well. Specially the Stop sign which has its unique shape.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| No Vehicle     		| No Vehicle  									|
| Turn Right Ahead		| Turn Right Ahead								|
| Road Work				| Road Work										|
| 50 km/h	      		| 60 Km/h					 				    |
| No Passing			| No Passing    						|

The model was able to guess the 4 out of 6 correctly which given the accuracy of `83%`. It is worse than the accuracy I got with the training set which was `97%`.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image8]
![alt text][image9]

As It can be seen in the images above, the softmax prediction are all correct wit
 100%. In the next two image which were predicted wrongly we will go through more details.

![alt text][image7]

It can be seen that the prediction was 58% for 60Km/h and only 42% for 50Km/h which was the correct guess.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][image10]

The biggest prominent characteristics used by the network, would be the edges, As it can be seen in the image, (Which how the stop sign is learned), The edges outside and inside of the image are used. It can also be seen that the shape of the given sign plays a significant role, The color of the sign would also be important.