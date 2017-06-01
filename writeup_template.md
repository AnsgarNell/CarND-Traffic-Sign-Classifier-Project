#**Traffic Sign Recognition** 
---


[//]: # (Image References)

[image1]: ./writeup_images/classes.png "Classes"
[image2]: ./writeup_images/image_original.png "Original"
[image3]: ./writeup_images/image_normalized.png "Normalized"
[image4]: ./web_images/80.jpg "Traffic Sign 1"
[image5]: ./web_images/240_F_148528289_8FbyCvR8JGsP5ZBf8BszYwiVtrhjyzPx.jpg "Traffic Sign 2"
[image6]: ./web_images/adelanto.jpg "Traffic Sign 3"
[image7]: ./web_images/ceda.jpg "Traffic Sign 4"
[image8]: ./web_images/obra.jpg "Traffic Sign 5"
[image9]: ./web_images/paso_prohibido.jpg "Traffic Sign 6"
[softmax1]: ./softmax/12.png "Traffic Sign 12"
[softmax2]: ./softmax/5.png "Traffic Sign 5"
[softmax3]: ./softmax/9.png "Traffic Sign 9"
[softmax4]: ./softmax/13.png "Traffic Sign 13"
[softmax5]: ./softmax/14.png "Traffic Sign 14"
[softmax6]: ./softmax/17.png "Traffic Sign 17"
[feature0]: ./featureMaps/0.png "Feature Map 0"
[feature1]: ./featureMaps/1.png "Feature Map 1"
[feature2]: ./featureMaps/2.png "Feature Map 2"
[feature3]: ./featureMaps/3.png "Feature Map 3"
[feature4]: ./featureMaps/4.png "Feature Map 4"
[feature5]: ./featureMaps/5.png "Feature Map 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/AnsgarNell/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

Or the final version in [HTML code](http://htmlpreview.github.io/?https://github.com/AnsgarNell/CarND-Traffic-Sign-Classifier-Project/blob/master/report.html "HTML code")

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how much samples the set contains for each class.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after normalizing with the formula (pixel - 128)/ 128.

![alt text][image2]
![alt text][image3]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description								| 
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   					|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6	 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16	 				|
| Flatten				| input 5x5x16, output 400      				|
| Fully connected		| input 400, output 120     					|
| RELU					|												|
| Dropout				| Rate 70%										|
| Fully connected		| input 120, output 84	     					|
| RELU					|												|
| Dropout				| Rate 70%										|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used 20 epochs as tests with higher values didn't show big increases in accuracy and do to the time consuming it implies without a CUDA ready GPU.

For the batch size I used the default 128 value.

The selected optimizer and other parameters where taken with the same values as in the LeNet Lab example.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.938
* test set accuracy of 0.929

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen? The first architecture was the same as in the LeNet example. Normalizing was done to RGB images with the (pixel_value - 128)/128 formula.


* What were some problems with the initial architecture? Accuracy was below 0.9.
* How was the architecture adjusted and why was it adjusted? First grayscaling was done changing the input from RGB to just one channel. Adjustments where made in the model input and x placeholder to reflect this change.  Then two dropout layers where introduced after both fully connected layers as suggested by forum mentors.

* Which parameters were tuned? How were they adjusted and why? The epoch parameter was changed, beginning with 10, and ending with 20 after testing the values 50 and 30. Dropout keep probability parameter was also tested with a value of 0.5 but results where not so good.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:

* What architecture was chosen? Lenet

* Why did you believe it would be relevant to the traffic sign application? Because it was the one used in the examples and suggested to begin with.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? Accuracy is over 90% in all cases, being the lowest (test accuracy) 92.9% which is a quite good result.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web (can be also found in the web_images folder):

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]

The first image might be difficult to classify because the number 8 may sometimes be taken for a 5 (this happened several times, but not in this training session).

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
| Priority road	      	| Priority road			 						|
| 80 km/h	      		| 80 km/h				 						|
| No passing      		| No passing   									|
| Yield 	      		| Yield 	   									|
| Stop      			| Stop  										|
| No entry				| No entry										|


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 92.9%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![alt text][softmax1]

For the first image, the model is very sure that this is a Priority road sign. The first choice's probability is almost 1, and the others are near to zero.

![alt text][softmax2]

In this case, the system has correctly detected a speed limitation sign. It's first choice is the real one, 80 km/h. The next two options, 60 km/h and 50 km/h, are really reasonable as the numbers' shape is similar. In fact, in other training sessions this sign was not correctly classified and was confused with the 50 km/h one.

![alt text][softmax3]

This sign was also problematic in some sessions. Its probability isn't as high as the precedent ones, being 0.655. The next possibilities that were taken into account by the architecture were sign number 16 (Vehicles over 3.5 metric tons prohibited) and 10 (No passing for vehicles over 3.5 metric tons) which makes sense as they are similar.

The next signs (and also the first, Priority road) have a particular shape which makes them very easy to classify, as can be seen in the probabilities results provided by the softmax function:

![alt text][softmax4]
![alt text][softmax5]
![alt text][softmax6]


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

This is the features map for the first case:

![alt text][feature0]

As it can be seen, the network based its classification in the rhombus shape of the sign, which is the most significant characteristic.

![alt text][feature1]

In the 80 km/h speed limit sign the most important things that we can see are the circle and the numbers, although in some images it isn't clear if the first digit is an eight or a six.

![alt text][feature2]

The No passing picture has again a characteristic circle, and two not very clear figures in the middle, which lead to the low probability for this sign.

![alt text][feature3]

In this case, we can see a Yield sign, and two different features: an inverted triangle, and also the white (or black) figure in the middle of it.

![alt text][feature4]

For the Stop sign we can quite clearly read the STOP letters, and we can also see it as a octagon.

![alt text][feature5]

Finally, the most eye-catching characteristic is the big horizontal line in the middle of the sign.