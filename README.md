# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./test_images/index.png "Visualization"
[image2]: ./test_images/visualization2.png "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_images/20953895-bicycle-crossing-traffic-warning-sign-diamond-shaped-traffic-signs-warn-drivers-of-upcoming-road-con.jpg "Traffic Sign 1"
[image5]: ./test_images/459381023.jpg "Traffic Sign 2"
[image6]: ./test_images/459381273.jpg "Traffic Sign 3"
[image7]: ./test_images/459381295.jpg "Traffic Sign 4"
[image8]: ./test_images/mifuUb0.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained through the second code cell of the IPython notebook into the twelvth one.  

I used the NumPy library to calculate summary statistics of the traffic
signs data set:

* Grouping images by 'Tracks'.
* Comparing similarities between consecutive images in the training set and testing set.
* Distribution of the images in different classes which are 43

Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in 10:12 code cells of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

I tried several colorspaces, starting from original RGB followed by YUV where Y channel carries intensity information which was normalized for each image to adjust for variable lighting. I found that Y channel was almost identical to grayscale. However, this naive linear scaling is not always sufficient as evident in figure above where contrast adjustment makes a huge difference. After trial and error, I decided to use contrast limited adaptive histogram equilization (CLAHE) with tile size of 4x4. I used all color channels as this information is relevant for traffic signs. As a last step, I simply scaled RGB values to lie in the range [-0.5,0.5]. Note that actual pre-processing is applied later on in the code after data augmentaion.

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the 14 code cell of the IPython notebook. 

As already explained, images in the same track are highly correlated and randomly distributing them between training and validation set defeats the purpose of validation. So, I selected one track (consisting of 30 images) per class for validation set. Although this is far from ideal in terms of size of validation set, it is still better that mixing highly correlated images between training and validation set. This still leaves way too many correlated images and at the same time not enough images in the training set. To resolve this problem, I augmented dataset by performing random scaling (range of [0.85,1.1]), rotation (+/- 17 degrees) and translation (+/- 2 pixels). Further, I also perturbed every consecutive image in the training set, idea being that any two consecutive images are highly correlated. To balance training dataset, I augmented dataset so that there are 5000 images per class. Note that besides image pre-processing (contrast-normalization), the validation and test sets were neither augmented, nor perturbed.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 23 cell of the ipython notebook.

The model I have chosen is inspired by VGGnet architecture. Consider the following arrangement :

* Conv. layer
* ReLu activation
* Conv. layer
* ReLu activation
* Pooling
* Dropout

This arrangement is repeated 3 times so that we have a total of 6 conv layers. These layers have 32, 32, 64, 64, 128 and 128 filters in that order. Each conv. layer uses 3x3 filters with stride=1 and padding=1. Pooling layer uses maxpool filters of size 2x2 with stride=2. This is followed by following arrangement :

* Full-connected layer
* ReLu activation
* Dropout

repeated twice. Each fully-connected layers has size of 128. Softmax function is applied on final output layer for computing loss.


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 24 cell of the ipython notebook. 

I used Adam optimizer with learning rate = 0.007. Batch size of 64 was used and training was done for 30 epochs. The keep_prob for dropout layers was chosen to be 0.5 for conv layers and 0.7 for FC layers.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 25 cell of the Ipython notebook.

My final model results were:
* test set accuracy of 98.2%


1. Starting with pre-processing stage, I tried several color channels including RGB, YUV and RGB with CLAHE. I chose RGB with CLAHE as it gave best results and I believe color carries useful information for our problem.
2. I chose architecture similar to VGGnet because deeper layers give better results and it is also quite elegant in the sense that same size filters for conv layers and pooling are used throughout.
3. Padding=1 or 'SAME' padding was used in conv layers to retain information at the borders of images.
4. Batch size and the size of FC layers was mostly constrained by memory issues, and adding more neurons in FC layers didn't seem to help much with the accuracy.
5. keep_prob in dropout for FC layers was chosen to be 0.7 because a smaller value of ~0.5 led to extremely slow convergence.
6. I used Adam optimizer as it seems to automatically tune the learning rate with time.

 An architecture similar to VGGnet the well known architecture was chosen:
 
 * An architecture similar to VGGnet was chosen.
 * The VGGnet is a deep convolutional neural network for object recognition (our very same problem here) developed and trained by Oxford's renowed Visual Geometry Group which achieved very good performance on the ImageNet dataset. In my opinion, VGGNet extracts features that are slightly more general and more effective for datasets other than ILSVRC. I did tests on the dataset I'm working on, and they confirmed it.
 * Final test set accuracy is 98.2% which is very well. I think it is worth it in that scenario as explained above.



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image of bicyecles might be difficult to classify because it is not the exact one that the neural network trained on from German signs.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 29 cell of the Ipython notebook.

 I get an accuracy of 80% on the new dataset. This is actually better that what I had anticipated, my classifier worked even on the Bicycle image which looked very different from the German sign.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 30-31 cell of the Ipython notebook.

![alt text][image2]

For the Bicycle sign, classifier is very uncertain with the best prediction because it is very different than German set. The german road priority sign is the same with Bicycle sign present here in two big aspects (geometric shape, Color) While the Bicycle sign has only the Bicycle drawing (less than 10% of picture) so the classifier based on its own training experience it is very certain that it is a priority road sign not a Bicycle one so it did well in trying to capture the sign. The Bicycle sign he knows well has a different geometric shape and color.
