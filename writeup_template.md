## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeUpImages/01_vechiles_samples.png
[image2]: ./writeUpImages/02_non_vechiles_samples.png
[image3]: ./writeUpImages/03_car_hog.png
[image4]: ./writeUpImages/04_non_car_hog.png
[image5]: ./writeUpImages/05_find_cars.png
[image6]: ./writeUpImages/06_possible_coverage_x10.png
[image7]: ./writeUpImages/07_possible_coverage_x15.png
[image8]: ./writeUpImages/08_possible_coverage_x20.png
[image9]: ./writeUpImages/09_possible_coverage_x25.png
[image10]: ./writeUpImages/10_heat_map.png
[image11]: ./writeUpImages/11_heat_map_thresholded.png
[image12]: ./writeUpImages/12_heat_map_SciPyLabel.png
[image13]: ./writeUpImages/13_final_output1.png
[image14]: ./writeUpImages/14_final_output2.png
[image15]: ./writeUpImages/15_1.png
[image16]: ./writeUpImages/15_1_done.png
[image17]: ./writeUpImages/15_2.png
[image18]: ./writeUpImages/15_2_done.png
[image19]: ./writeUpImages/15_3.png
[image20]: ./writeUpImages/15_3_done.png
[image21]: ./writeUpImages/15_4.png
[image22]: ./writeUpImages/15_4_done.png
[image23]: ./writeUpImages/15_5.png
[image24]: ./writeUpImages/15_5_done.png
[image25]: ./writeUpImages/15_6.png
[image26]: ./writeUpImages/15_6_done.png
[image27]: ./writeUpImages/05_2_find_cars.png

[video1]: ./test_video_output.mp4
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is defined at line 39 in file `CAR_CLASSIFIER.py`, and called by:

```python
oClassifier.extract_data_features({
    'color_space': 'YUV',
    'spatial_size': (16, 16),
    'hist_bins': 16,
    'orient': 9,
    'pix_per_cell': 8,
    'cell_per_block': 2,
    'hog_channel': 'ALL'
})
```

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  
I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image3]
![alt text][image4]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found this setting to give me most optimal result for the final output: 

```python
oClassifier.extract_data_features({
    'color_space': 'YUV',
    'spatial_size': (16, 16),
    'hist_bins': 16,
    'orient': 9,
    'pix_per_cell': 8,
    'cell_per_block': 2,
    'hog_channel': 'ALL'
})
```

#### 3. Data Normalization

Before training the data, I normalized the training data by using `StandardScaler` function from scikit-learn, then split the data into training and testing data

#### 4. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear Support Vector Machine Classifier (SVC) based on function SVM from scikit-learn. 
Again, as shown above, I used `ALL` channels of images and in `YUV` corlor space

For the code, the function for training the classifier was defined at line 94 in file `CAR_CLASSIFIER.py` and called by:

```python
oClassifier.train_SVC()
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  
How did you decide what scales to search and how much to overlap windows?

The code for sliding window search is defined at line 163 in file `CAR_CLASSIFIER.py`.


I tried different scales of search windows from around the horizon of the image to the bottom of the image since these are the locations where cars would appear to be.
Here is the result:

![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]

These settings gives a pretty decent coverage. Note that if the scale is less than 1 for the 8x8 search window, the result would appear to be less accurate.
Thus I chose 1 as base value for scale, then gradually increased it to cover the entire region of interests


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here is an example of detected result through sliding window search:

![alt text][image5]

I tried different cell steps by using `CAR_CLASSIFIER.set_sliding_window_param()`, and I found by setting the cell step to 1 gives the best detection later on for the heatmap.
And I tuned the search range a little bit to make sure to maximize number of search windows possible

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

(The template pretty much covered what I did....)

I recorded the positions of positive detections in each frame of the video. 
From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  
I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  
I then assumed each blob corresponded to a vehicle.  
I constructed bounding boxes to cover the area of each blob detected.  

Here is the processing pipeline:

1. Slinding windows search:

![alt text][image27]

2. Generating heatmaps by detected box superposition:

![alt text][image10]

3. Since detected cars will have high numbers of detected box overlap, thus applying threshold values to the heatmap to eliminate the false positives:

![alt text][image11]

4. Lastly, use `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. 

![alt text][image12]

5. And here is the final output

![alt text][image14]


### Here are six frames and their corresponding processed images:

![alt text][image15]
![alt text][image16]

![alt text][image17]
![alt text][image18]

![alt text][image19]
![alt text][image20]

![alt text][image21]
![alt text][image22]

![alt text][image23]
![alt text][image24]

![alt text][image25]
![alt text][image26]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  
Where will your pipeline likely fail?  What could you do to make it more robust?

The final output still has some artifacts, and that is fully expected. 
Some false positives still remain after heatmap thresholding (in later half of the video)
To over come this I believe we need more data, much more than what we have to train a robust model. 
