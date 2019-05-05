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