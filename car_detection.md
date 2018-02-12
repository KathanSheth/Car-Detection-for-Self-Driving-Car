## Vehicle Detection

---



The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car.png
[image2]: ./examples/noncar.png
[image3]: ./examples/car_hog.png
[image4]: ./examples/noncar_hog.png
[image5]: ./examples/multiple_frames.png
[image6]: ./examples/multiple_frames_2.png
[image7]: ./examples/result.png
[image8]: ./examples/heatmap.png

[video1]: ./project_video.mp4


---
### Writeup / README
First, I have used training data provided by Udacity which includes around 8792 car images and 8968 non car images.

### Histogram of Oriented Gradients (HOG)

#### 1. HOG Feature Extraction

I have used `get_hog_features` function to extract HOG feature of an image.
I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

Below are random images for car and noncar each:

![alt text][image1]
![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed above images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

I finally decided to go with following choice of parameters:
Orinetation = 8
pixels_per_cell = 8
cells_per_block = 2

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image3]
![alt text][image4]


#### 2. Classifier

I trained a linear SVM using HOG, color and spatial features. I have used following parameters to extract the features from vehicle and non vehicle data.

color_space = 'YCrCb'

spatial_size = (16, 16)

orient = 8

pix_per_cell = 8

cell_per_block = 2

hog_channel = 'ALL'

hist_bins = 32

spatial_feat=True 

hist_feat=True

hog_feat=True

I used 'StandardScaler' function for normalization and `train_test_split` of sklearn to divide data in test and train dataset.

### Sliding Window Search

#### 1. Sliding Window

I have used following parameters for sliding window. It is written in `sliding_window` function.

Window size = 64
Overlap = 0.85

I have used `search_windows` functions to get windows for positive detection. 

Following are some examples for sliding window and search window functions.

![alt text][image5]
![alt text][image6]

### False Positives

As seen from above images, I am detecting many false positives for cars. To remove this, I have heatmap and then thresholded that map to identify correct vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Also, to detect car for different distances and variable window sizes, I am using different combinations of ystart,ystop and scale values.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][image7]


### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap frame:

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image8]


---

### Video Implementation

Finally, I have combined all functionalities in `process_image` function and then created my video.

Here's a [link to my video result](./output_video/project_video.mp4)



---

### Discussion

Here I have used Linear SVM concept. I feel that concept of Neural Network can be implemented and can work better. Also, I have done a lot trial and error to detect and draw variable window size which can be improvized with some more tweaking or with some good idea. (Black fox for me!!). 

