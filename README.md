---
typora-copy-images-to: assets

---

## Advanced Lane Finding using OpenCV
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive) <img src="https://engineering.nyu.edu/sites/default/files/2019-01/tandon_long_color.png" alt="NYU Logo" width="130" height="whatever">

![Lanes Image](./examples/example_output.jpg)

The following project is a part of Udacity’s Self Driving car engineering NanoDegree program. The aim of project is to successfully find the radius of curvature as well as the vehicle offset from the lane.

The Project
---

The steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position. 

## The folder structure

| Name of Folder     | Contains                                      |
| ------------------ | --------------------------------------------- |
| Assets             | Resources for README                          |
| camera_cal         | Input images for camera calibration           |
| undistorted_images | Undistorted images after camera calibration   |
| test_images        | Test images for pipeline                      |
| test_images_output | Output images after image thresholding        |
| video_input        | Test videos for the final pipeline            |
| video_output       | Test video output (6 seconds each)            |
| video_output_final | Final output video folder (Full length video) |

## Prerequisites

- pip 
- python 3
- virtual environment

## Install instructions

`open terminal`

```bash
$ git clone https://github.com/Shivam-Bhardwaj/AdvanceLaneFinding.git
$ virtualenv --no-site-packages -p python3 venv 
$ source venv/bin/activate
$ cd AdvanceLaneFinding
$ pip install -r requirements.txt
$ jupyter notebook
```

`open FinalCode.ipnyb`

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

#### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

------

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.    

The README.md file is an extensive writeup of the project. It includes the code folder architecture, resources, test output, Jupyter Notebook etc. Please contact 

Shivam Bhardwaj 

 [LinkedIn](<https://www.linkedin.com/in/shivamnyu/>) [Instagram](https://www.instagram.com/lazy.shivam/) [Facebook](<https://www.facebook.com/shivambhardwaj2008>) 

Mail to shivam.bhardwaj@nyu.edu

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The Camera calibration is taken care under the section of **Calibrating the camera**

I started by reading a random calibration image from `camera_cal` folder to get the parameters of the images.

Using the following code snippet:

```python
img = mpimg.imread('camera_cal/calibration11.jpg')
image_shape = img.shape

# nx and ny are taken as 9 & 6 respectively to denote the number of squares in the image.

nx = 9
ny = 6
```

I start by preparing `"object points"`, which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  

Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Distorted vs undistorted image](assets/image1-1557120372283.png)

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Using the distortion correction parameters obtained above I used the follwing line to get an undistorted image shown below:
![](assets/image2-1557120434334.png)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The section **Declaring functions important for Gradients and Color transforms** has the functions for performing different image transforms and masks. The code significantly self explanatory. However, they are explained briefly below:

- `get_thresholded_image(img)` Function to do the undistortion, conversion to grayscale and creating a mask based on pixel threshold

  - `cv2.undistort(img, cameraMatrix, distortionCoeffs, None, cameraMatrix)`
  - `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)`
  -  `mask = np.zeros_like(color_combined)`

- `abs_sobel_thresh(gray, orient='x', thresh_min=0, thresh_max=255)` 

  - ```python
     if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    ```

- `dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2))`

The following image is an example when the above filters are applied.

![](assets/Screenshot from 2019-05-06 02-45-18.png)

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

To obtain the perspective transform OpenCV's `cv2.warpPerspective(thresholded, M, img_size , flags=cv2.INTER_LINEAR)` function is used. For which the source and destination points are chosen as explained below.

```python
# Vertices extracted manually for performing a perspective transform

bottom_left = [200,720]
bottom_right = [1110, 720]
top_left = [570, 470]
top_right = [722, 470]
```

```python
source = np.float32([bottom_left,bottom_right,top_right,top_left])
```

```python
# Destination points are chosen such that straight lanes appear more or less parallel in the transformed image.

bottom_left = [320,720]
bottom_right = [920, 720]
top_left = [320, 1]
top_right = [920, 1]
```

This resulted in the following source and destination points:

| Point        |  Source   | Destination |
| ------------ | :-------: | :---------: |
| bottom_left  |  200,720  |   320,720   |
| bottom_right | 1110, 720 |  920, 720   |
| top_left     | 570, 470  |   320, 1    |
| top_right    | 722, 470  |   920, 1    |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

After performing color thresholding:

![](assets/image2-1557125794339.png)

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To get the final second degree polynomial for the lanes, I had to perform multiple steps.

To begin with, a histogram was created from the bottom half of the image to get the lane starting as shown below.

```python
histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)

# Peak in the first half indicates the likely position of the left lane

half_width = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:half_width])

# Peak in the second half indicates the likely position of the right lane

rightx_base = np.argmax(histogram[half_width:]) + half_width

plt.plot(histogram)
print("The base of the lines are for LEFT LINE:",leftx_base,"pixels & for Right Line:", rightx_base, "pixels")
```

![Histogram](assets/image.png)

Then the whole image was divided into 10 windows to perform a sliding window search.

```python
num_windows = 10
num_rows = warped.shape[0]
window_height = np.int(num_rows/num_windows)
window_half_width = 70
```

A sliding window search is done per window in vertical direction. If there is a significant change in the sum of pixels in vertical direction the bounding box`(represented in green)` is shifter per consecutive window. Till it is done on the whole image. 

**NOTE: This is only performed on first frame.**

![](assets/image-1557126862324.png)

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

------

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

------

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

- 

[//]: #	"Image References"
[image1]: ./camera_cal/calibration1.jpg	"Undistorted"
[image2]: ./test_images/test1.jpg	"Road Transformed"
[image3]: ./examples/binary_combo_example.jpg	"Binary Example"
[image4]: ./examples/warped_straight_lines.jpg	"Warp Example"
[image5]: ./examples/color_fit_lines.jpg	"Fit Visual"
[image6]: ./examples/example_output.jpg	"Output"
[video1]: ./project_video.mp4	"Video"



