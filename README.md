# Udacity Project 4: Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


This project builds on the [first project](https://github.com/stefancyliax/CarND-LaneLines-P1), but now uses advanced method of finding lines. In the end the implemented pipeline was able to calmly find and follow the lines in the provided video. This time we also calculated the curvature of the street and the offset of the vehicle from the middle of the road. This document describes my approach and implemented pipeline.

The code for the project can be found in the [Jupyter Notebook](https://github.com/stefancyliax/CarND-Advanced-Lane-Lines/blob/master/P4_Advanced_Lane_Lines.ipynb).

The pipeline consists of the following steps.
1. Undistort image
2. Get grayscale image by dropping G and B channel of RGB image
3. Do perspective transformation for get "bird-view" of street
4. Apply dynamic sliding window thresholding (see below for explanation of algorithm)
5. Find lane lines left and right, calculate lane line curvature
6. Calculate offset from middle of the street
7. Compose resulting image with marked lane lines, curvature and offset

![pipeline](https://github.com/stefancyliax/CarND-Advanced-Lane-Lines/raw/master/output_images/pipeline1.png)

![result](https://github.com/stefancyliax/CarND-Advanced-Lane-Lines/raw/master/output_images/result.png)

Project video: (Youtube link)

[![Project track](https://github.com/stefancyliax/CarND-Advanced-Lane-Lines/raw/master/output_images/Y62xDK.gif)](https://youtu.be/g2TaB6dh0DY)


Video with debugging output: (Youtube link)

[![Project track](https://github.com/stefancyliax/CarND-Advanced-Lane-Lines/raw/master/output_images/VmYrq1.gif)](https://youtu.be/edLIRW_hTIc)

---

## Detailed explanation of the pipeline.


### 1. Distortion Correction
Every camera's optical system has certain characteristic, that lead to a distortion of the image. These distortion can be measured and removed. For this we were provided with a set of 20 calibration images showing a chessboard pattern. Using the OpenCV functions `cv2.findChessboardCorners` and `cv2.calibrateCamera` we calculated the camera matrix, distortion coefficients, rotation and translation vectors.

![undistort sample 1](https://github.com/stefancyliax/CarND-Advanced-Lane-Lines/raw/master/output_images/undistort1.png)

![undistort sample 2](https://github.com/stefancyliax/CarND-Advanced-Lane-Lines/raw/master/output_images/undistort2.png)

```python
def get_camera_calibration_parameters(globdir='./camera_cal/calibration*.jpg'):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(globdir)

    # Step through the list and search for chessboard corners
    for fname in images:
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    assert len(objpoints) == len(imgpoints) # Check if same number of object and image point found

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return ret, mtx, dist
```
With these we could undistort an image using the function `cv2.undistort` as a first step in the pipeline.

```python
def undistort_img(img, mtx=mtx, dist=dist):
        return cv2.undistort(img, mtx, dist, None, mtx)
```

### 2. Get grayscale image by dropping G and B channel of RGB image
[Since a simple grayscale conversion loses all contrast for the yellow lines](https://github.com/stefancyliax/CarND-LaneLines-P1#grayscale-conversion), I used the red channel exclusively.

I did comprehensive tests with different color spaces, e.g. S from HSV and B from LAB but none provided significantly better results than the red channel.

```python
def get_redchannel(img):
    """
    Helper function to extract the R channel from RGB color space
    """
    red = img[:,:,0].copy() ## use red channel
    return red
```
![red channel](https://github.com/stefancyliax/CarND-Advanced-Lane-Lines/raw/master/output_images/r_chan.png)


### 3. Do perspective transformation for get bird’s-eye view of street
In the next step I did a perspective transformation to get to a bird’s-eye view. This results in parallel lane lines and allows a calculation of the line curvature.

The source and destination points for the transformation were chosen by hand using straight_lines2.jpg.

```python
def get_warp_parameters():
"""
Helper function to calculate transformation matrix
"""
  src = np.float32(
      [(580, 450),
       (700, 450),
       (1130, 680),
       (150, 680)])

  dst = np.float32(
      [(450, 200),
       (830, 200),
       (830, 720),
       (450, 720)])

  M = cv2.getPerspectiveTransform(src, dst)
  Minv = cv2.getPerspectiveTransform(dst, src)
  return M, Minv

def warp(img, M=M):
    """
    Helper function to apply perspective warp to an image
    """
    return cv2.warpPerspective(img, M, (1280,720), flags=cv2.INTER_LINEAR)
```
![warp sample](https://github.com/stefancyliax/CarND-Advanced-Lane-Lines/raw/master/output_images/warp.png)


### 4. Apply dynamic sliding window Thresholding
[After comprehensive tests with static thresholding, dynamic thresholding, sobel parameters](https://github.com/stefancyliax/CarND-Advanced-Lane-Lines/blob/master/P4_Advanced_Lane_Lines.ipynb#Thresholding) I implemented a dynamic thresholding that slices the image in 20 horizontal sections and selects the brightest 1% of pixels in each slice. This provided to be very powerful when applied on an already warped image.

```python
def apply_threshold_dynamic_sliding_window(img, percentile=0.015):
    """
    Slice the image in horizontal sections and do dynamic thresholding on each slice.
    """
    binary_output = np.zeros_like(img)
    percentile = 100 - (percentile*100)

    nb_slice = 20 # number of slices to be used
    slice_height = np.int(img.shape[0]/nb_slice)

    # loop over slices and calculate dynamic threshold
    for n in range(nb_slice):
        slice_window = img[n*slice_height:(n+1)*slice_height, :]    
        lower_thresh = np.percentile(slice_window, percentile)
        binary_output[n*slice_height:(n+1)*slice_height, :][(slice_window >= lower_thresh)] = 1

    return binary_output
```
![Thresholding Sample 1](https://github.com/stefancyliax/CarND-Advanced-Lane-Lines/raw/master/output_images/thresh1.png)
![Thresholding Sample 2](https://github.com/stefancyliax/CarND-Advanced-Lane-Lines/raw/master/output_images/thresh2.png)

### 5. Find lane lines left and right, calculate lane line curvature
To find the lane lines in the thresholded images I implemented a class. I chose to implement the class for a single lane line to reduce the redundant code. After instantiating objects for the left and right line, I could pass the current image to it, using the `Line.process_img()` method.
The class would then:
1. Find the pixel belonging to a lane line
2. Fit a second second order polynomial
3. Sanity check the fit against previous fits
4. Smooth over 10 previous fits
5. Calculate the lane line curvature

```python
def process_img(self, img):        
    # assure that no color images are passed
    assert img.ndim == 2

    # if there are no prior lines detected, start sliding window algorithm
    if self.nb_detected_lines == 0:

        # discard oldest part of pool if there were several lines that didn't pass the sanity check
        self.fit_pool = self.fit_pool[3:,:]
        self.fit_cr_pool = self.fit_cr_pool[3:,:]

        # if there are fits left in the pool, update fit_smooth for use in later sanity check
        if self.fit_pool.size > 0:
            self.fit_smooth = np.mean(self.fit_pool, axis=0)            

        # Prepare sliding window algorithm for lane line finding.
        # For left line calculate histogram on left half of the image, for right line on right half.
        histogram = np.sum(img[360:,self.window_start:self.window_start+290], axis=0)
        peak = np.argmax(histogram) + self.window_start

        # Do lane line finding with sliding window algorithm
        fit, fit_cr = self.__find_line_sliding_window(img, peak)

    # otherwise do simple line finding based on current average fit
    else:
        fit, fit_cr = self.__find_line_simple(img, self.fit_smooth)

    # Do sanity check on the newly found fit parameters. If they pass update parameters.
    if self.__sanitycheck_parameters(fit, fit_cr):     
        self.__update_parameters(fit, fit_cr)
```
#### 5.1 Find the pixels and fit
For the line finding two algorithms are implemented, `Line.__find_line_sliding_window()` and `Line.__find_line_simple()`.

`Line.__find_line_sliding_window()` uses a histogram to find a starting point for the sliding windows at the bottom of the image. The sliding window is then moved up in 9 steps, adjusting the center to the mean of the found pixel. All found pixel are then passed to `numpy.polyfit()` to find a second order polynomial.
```python
def __find_line_sliding_window(self, img, peak):
    nwindows = 9
    window_height = np.int(img.shape[0]/nwindows)
    margin = 75
    nb_pixel_to_adjust_window_pos = 50 # number of pixels to adjust position of window
    lane_coordx = np.array([])
    lane_coordy = np.array([])

    for n in range(nwindows):
        # slice image into horizontal slice
        window_lower_border = img.shape[0] - n*window_height
        window_slice = img[window_lower_border-window_height:window_lower_border, :]

        # find all white pixels within margin of peak
        slice_window = window_slice[: , peak-margin:peak+margin]
        slice_line_coordy, slice_line_coordx = np.nonzero(slice_window)
        # compensate offset of coordinates
        slice_line_coordy += window_lower_border - window_height
        slice_line_coordx += peak - margin

        # append found pixels
        lane_coordx = np.append(lane_coordx, slice_line_coordx)
        lane_coordy = np.append(lane_coordy, slice_line_coordy)

        # recenter window if more than (adjust_window_pos) pixel are found
        if slice_line_coordx.size > nb_pixel_to_adjust_window_pos:
            peak = np.int(np.mean(slice_line_coordx))

    # calculate fit and scaled fit
    fit_cr = np.polyfit(lane_coordy*ym_per_pix, lane_coordx*xm_per_pix, 2)
    fit = np.polyfit(lane_coordy, lane_coordx, 2)

    return fit, fit_cr
```
![Line finding sliding window 2](https://github.com/stefancyliax/CarND-Advanced-Lane-Lines/raw/master/output_images/find_slide.png)

`Line.__find_line_simple()` uses an existing polynomial to find pixel within a certain margin from it. Again the found pixel are then passed to `numpy.polyfit()`.

```python
def __find_line_simple(self, img, fit_old):
    margin = 50
    lane_coordx = np.array([])
    lane_coordy = np.array([])

    # find all white pixels in image
    px_line_coordy, px_line_coordx = np.nonzero(img)
    px_line_coord = np.transpose(np.asarray(np.nonzero(img)))

    # find all white pixels within margin from last fit
    test = fit_old[0]*(px_line_coord[:,0]**2) + fit_old[1]*px_line_coord[:,0] + fit_old [2]
    lane_coord = px_line_coord[((test+margin) > px_line_coord[:,1]) & ((test-margin) < px_line_coord[:,1])]

    lane_coordy = lane_coord[:,0]
    lane_coordx = lane_coord[:,1]

    # calculate fit and scaled fit
    fit_cr = np.polyfit(lane_coordy*ym_per_pix, lane_coordx*xm_per_pix, 2)
    fit = np.polyfit(lane_coordy, lane_coordx, 2)

    return fit, fit_cr
  ```
![Line finding simple](https://github.com/stefancyliax/CarND-Advanced-Lane-Lines/raw/master/output_images/find_simple.png)

#### 5.2 Sanity check
The sanity check is done by comparing the found polynomial parameters to previously found ones and assuring, that they didn't change more than 30%. The relatively high variation of 30% is necessary because the second order polynomial parameter of a straight line are near 0. It's not a perfect solution but works well enough.
```python
def __sanitycheck_parameters(self, fit, fit_cr):
    # if fit pool is empty just save new fit, since we can't do any sanity checks
    if self.fit_pool.size == 0:
        self.nb_detected_lines = 1
        self.fit_valid = True
        return True

    # check if new fit in between +-30% of the current average fit
    diff = np.abs(fit - self.fit_smooth)
    if np.all(diff < np.abs(self.fit_smooth)*0.3):
        if self.nb_detected_lines < 10: # bound to 10
            self.nb_detected_lines += 1
        self.fit_valid = True
        return True

    else:
        if self.nb_detected_lines > 0:
            self.nb_detected_lines -= 1
        self.fit_valid = False
        return False

    assert False
```


#### 5.3 Smoothing and calculation the lane line curvature
To smoothen the drawn lines a pool of the 10 last polynomial parameters is kept and an average calculated.

The curvature of the lane lines is calculated at the very bottom of the image using the method from [this tutorial](http://www.intmath.com/applications-differentiation/8-radius-curvature.php). In it the first and second derivatives of the polynomial is used. The equation `curvature = ((1 + (2*A*y + B)**2)**1.5) / np.abs(2*A)` is used with A and B being the second and first order coefficients. `y` is set to 720 since we want the line curvature near the vehicle.

```python
def __update_parameters(self, fit, fit_cr):
    # update parameters with new found, valid fit
    self.fit_last = fit

    # append new valid fit to pool
    if self.fit_pool.size == 0:
        self.fit_pool = fit
        self.fit_pool = np.reshape(self.fit_pool, (1,3))
    else:
        self.fit_pool = np.vstack((self.fit_pool, fit))
        # only keep the last 10 fits
        if self.fit_pool.shape[0] > 10:
            self.fit_pool = self.fit_pool[1:,:]

    # average over last fits
    self.fit_smooth = np.mean(self.fit_pool, axis=0)

    # same for the scaled fit
    if self.fit_cr_pool.size == 0:
        self.fit_cr_pool = fit_cr
        self.fit_cr_pool = np.reshape(self.fit_cr_pool, (1,3))
    else:
        self.fit_cr_pool = np.vstack((self.fit_cr_pool, fit_cr))
        if self.fit_cr_pool.shape[0] > 10:
            self.fit_cr_pool = self.fit_cr_pool[1:,:]        

    # calculate lane line curvature
    y_eval = img.shape[0]
    self.curvature = ((1 + (2*np.mean(self.fit_cr_pool[:,0])*y_eval*ym_per_pix
                            + np.mean(self.fit_cr_pool[:,1]))**2)**1.5) / np.abs(2*np.mean(self.fit_cr_pool[:,0]))
```

### 6. Calculate offset from middle of the street
The offset is calculated from the two point were the polynomials intersect the lower border of the image and the middle of the image. Both intersections should have the same distance from the middle of the image for no offset from the middle of the lane. To calculate a potential offset, I averaged over both polynomial intersects and measure the distance to the middle of the image.
The intersections are calculated using `x = A*y^2 + B*y + C` with y=720 for the lower bottom of the image.

```python
def calculate_offset(left_fit, right_fit):
    y= 720
    xl = left_fit[0]* y**2 + left_fit[1]*y + left_fit[2]  
    xr = right_fit[0]* y**2 + right_fit[1]*y + right_fit[2]

    car_position = img.shape[1]/2
    lane_center_position = (xr + xl)/2
    offset_lane = (car_position - lane_center_position)*xm_per_pix
    return offset_lane
```


### 7. Compose resulting image with marked lane lines, curvature and offset
Finally visual displays of the detected lines were drawn back onto the input image.
To do this the polynomials were plotted on an empty canvas and then inversely warped using the transformation matrix `Minv`. Finally they were added to the input image.

```python
def draw_on_image(img, left_fit, right_fit):

    ploty = np.linspace(0, 719, num=720)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros(img.shape[0:2]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=12)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=12)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    return result
```

# Discussion

### Sobel operator
I chose not to used any Sobel operators in this project since I wasn't able to find sobel parameters that worked better than the color thresholding I used. I even experimented with diagonal sobel kernels with little success. I found that the sobel thresholds are very noisy.

For experimenting I build a [Ipython Interactive Widget for the sobel parameters.](https://github.com/stefancyliax/CarND-Advanced-Lane-Lines/blob/master/P4_Advanced_Lane_Lines.ipynb#Interactive-sobel-parameter-widget)


![sobel sample 1](https://github.com/stefancyliax/CarND-Advanced-Lane-Lines/raw/master/output_images/sobel1.png)

![sobel sample 2](https://github.com/stefancyliax/CarND-Advanced-Lane-Lines/raw/master/output_images/sobel2.png)

### Challenge Video
The pipeline isn't able to achieve good results on the challenge videos. It seems that this is mostly due to a slightly different camera orientation in the vehicle, that can be seen in a warped image. See below. Note that the warped image looks strange and the lane line are not even remotely parallel.

![challenge track](https://github.com/stefancyliax/CarND-Advanced-Lane-Lines/raw/master/output_images/challenge_track.png)

This would be easily handled with a custom transformation matrix. With it I'd expect pretty good results on the challenge video.

### Harder challenge video
My lane line finding algorithms fail completely on the harder challenge video mostly because the very tight corners that would require the sliding window algorithm to go sideways instead of steadily upwards.

![harder challenge track](https://github.com/stefancyliax/CarND-Advanced-Lane-Lines/raw/master/output_images/harder_challenge.png)

The very different lighting conditions are also a challenge.

It would be quite an interesting challenge to try to tackle it. Maybe I'll come back to it later.
