
# Advanced Lane Detection

Advanced Lane Detection consists of the following steps:
1. Camera calibration
2. Distortion correction
3. Color/gradient threshold
4. Perspective transform
5. Detect lane lines
6. Determine the lane curvature



## Camera Calibration
### Finding chessboard corners, Drawing detected corners on an image


```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
%matplotlib inline

def find_chessboard_corners():
    # prepare object points
    nx = 9 #TODO: enter the number of inside corners in x
    ny = 6 #TODO: enter the number of inside corners in y

    # Make a list of calibration images
    images = glob.glob('./camera_cal/calibration*.jpg')

    objpoints = [] #3d Real world space
    imgpoints = [] #2D image plane

    objp = np.zeros((ny*nx,3),np.float32)

    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) #x,y coordinates
    #fname = './camera_cal/calibration1.jpg'

    for fname in images:
        img = mpimg.imread(fname)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #plt.imshow(gray,'gray')
        #plt.show()
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        print (fname, ret)
        # If found, draw corners
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            #plt.imshow(img)
            #plt.show()
    return imgpoints, objpoints
```

### Camera calibration, Undistorting a test image


```python
def calibrateCamera(image, objpoints, imgpoints):
    # Read in an image
    #img = cv2.imread('calibration_test1.jpg')
    img = image
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # performs the camera calibration, image distortion correction and 
    # returns the undistorted image
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undistorted)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    return mtx, dist
```

## Perspective Transform

A perspective transform maps the points in a given image to different, desired, image points with a new perspective. The perspective transform you’ll be most interested in is a bird’s-eye view transform that let’s us view a lane from above; this will be useful for calculating the lane curvature later on. Aside from creating a bird’s eye view representation of an image, a perspective transform can also be used for all kinds of different view points.

### Undistort and Transform Perspective

    Undistort the image using cv2.undistort() with mtx and dist
    Convert to grayscale
    Find the chessboard corners
    Draw corners
    Define 4 source points (the outer 4 corners detected in the chessboard pattern)
    Define 4 destination points (must be listed in the same order as src points!)
    Use cv2.getPerspectiveTransform() to get M, the transform matrix
    use cv2.warpPerspective() to apply M and warp your image to a top-down view



## Gradient Threshold, Sobel Operator


```python
# Read in an image
im = cv2.imread('./test_images/test6.jpg')

#need to pass a single color channel to the cv2.Sobel() function, so first convert it to grayscale
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#Calculate the derivative in the x direction (the 1, 0 at the end denotes x direction):
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
#Calculate the derivative in the y direction (the 0, 1 at the end denotes y direction):
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
#Calculate the absolute value of the x derivative:
abs_sobelx = np.absolute(sobelx)
#Convert the absolute value image to 8-bit:
scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

thresh_min = 50
thresh_max = 100
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
plt.imshow(sxbinary, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7f441511a198>




![png](output_7_1.png)


## HLS and Color Thresholds
Here I'll read in the same original image (the image above), convert to grayscale, and apply a threshold that identifies the lines:


```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('./test_images/test6.jpg')
print('Original Image')
plt.imshow(image)
plt.show()
thresh = (180, 255)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
binary = np.zeros_like(gray)
binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1
print('Gray Thresold')
plt.imshow(binary, cmap='gray')
plt.show()
hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
H = hls[:,:,0]
L = hls[:,:,1]
S = hls[:,:,2]
print('H Gray')
plt.imshow(H, cmap='gray')
plt.show()
print('L Gray')
plt.imshow(L, cmap='gray')
plt.show()
print('S Gray')
plt.imshow(S, cmap='gray')
plt.show()
thresh = (90, 255)
binary = np.zeros_like(S)
binary[(S > thresh[0]) & (S <= thresh[1])] = 1
print('S binary')
plt.imshow(binary, cmap='gray')
plt.show()
```

    Original Image



![png](output_9_1.png)


    Gray Thresold



![png](output_9_3.png)


    H Gray



![png](output_9_5.png)


    L Gray



![png](output_9_7.png)


    S Gray



![png](output_9_9.png)


    S binary



![png](output_9_11.png)



```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Read in an image, you can also try test1.jpg or test4.jpg
image = mpimg.imread('./test_images/test6.jpg') 

# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output
    
hls_binary = hls_select(image, thresh=(90, 255))

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(hls_binary, cmap='gray')
ax2.set_title('Thresholded S', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
```


![png](output_10_0.png)


## Processing Each Image
### Thresholding

You'll want to try out various combinations of color and gradient thresholds to generate a binary image where the lane lines are clearly visible.

### Perspective Transform for straight lines

Next, you want to identify four source points for your perspective transform. In this case, you can assume the road is a flat plane. This isn't strictly true, but it can serve as an approximation for this project. You would like to pick four points in a trapezoidal shape (similar to region masking) that would represent a rectangle when looking down on the road from above.

The easiest way to do this is to investigate an image where the lane lines are straight, and find four points lying along the lines that, after perspective transform, make the lines look straight and vertical from a bird's eye view perspective

### Perspective Transform for curved lines

Those same four source points will now work to transform any image (again, under the assumption that the road is flat and the camera perspective hasn't changed). When applying the transform to new images, the test of whether or not you got the transform correct, is that the lane lines should appear parallel in the warped images, whether they are straight or curved.

Here's an example of applying a perspective transform to your thresholded binary image, using the same source and destination points as above, showing that the curved lines are (more or less) parallel in the transformed imag


```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#image = mpimg.imread('./test_images/straight_lines2.jpg')
#image = mpimg.imread('./test_images/test6.jpg')

# Edit this function to create your own pipeline.
def binary_threshold(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

def binary_threshold_undistort_warp(image, mtx, dist, src, dest):
    img_size = (1280,720) #image.shape
    #print(img_size)
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    #src = np.float32([(220,700), (585,450), (695,450), (1100,700)])
    #dest = np.float32([(220,700), (220,0), (1100,0), (1100,700)])

    #src = np.float32([(195,720), (580,460), (705,460), (1130,720)])
    #dest = np.float32([(320,720), (320,0), (960,0), (960,720)])

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dest)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(undist, M, img_size)

    cv2.line(image, (195, 720), (580, 460), (255,0,0), 2)
    cv2.line(image, (705, 460), (1130, 720), (255,0,0), 2)

    binary_warped = binary_threshold(warped)

    cv2.line(warped, (320, 720), (320, 0), (255,0,0), 2)
    cv2.line(warped, (960, 0), (960, 720), (255,0,0), 2)

    # Plot the result
    #f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    #f.tight_layout()

    #ax1.imshow(image)
    #ax1.set_title('Original Image', fontsize=40)

    #ax2.imshow(binary_warped,cmap='gray')
    #ax2.set_title('Pipeline Result', fontsize=40)
    #plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    #plt.show()

    #plt.imshow(warped)
    #plt.show()
    return undist, binary_warped
```

## Line Finding Method: Peaks in a Histogram

After applying calibration, thresholding, and a perspective transform to a road image, you should have a binary image where the lane lines stand out clearly. However, you still need to decide explicitly which pixels are part of the lines and which belong to the left line and which belong to the right line.

I first take a histogram along all the columns in the lower half of the image like this


```python
import numpy as np
def plot_hist(binary_warped):
    histogram = np.sum(binary_warped[:,:], axis=0)
    
    #plt.plot(histogram)
    #plt.show()
    return histogram
```

## Implement Sliding Windows and Fit a Polynomial

Suppose you've got a warped binary image called binary_warped and you want to find which "hot" pixels are associated with the lane lines. Here's a basic implementation of the method shown in the animation above. You should think about how you could improve this implementation to make sure you can find the lines as robustly as possible!


```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

def slide_window_fit_polynomial(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    #binary_warped = image
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    #plt.imshow(out_img)
    #plt.show()

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return out_img, left_fit, right_fit, lefty, leftx, righty, rightx
```

## Visualization of Sliding window

At this point, you're done! But here is how you can visualize the result as well:


```python
def plot_sliding_window(binary_warped, out_img, left_fit, right_fit, lefty, leftx, righty, rightx):
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    
    #plt.imshow(out_img)
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #plt.xlim(0, 1280)
    #plt.ylim(720, 0)
    #plt.show()
    return ploty, left_fitx, right_fitx
```

## Measuring Curvature

You're getting very close to a final result! You have a thresholded image, where you've estimated which pixels belong to the left and right lane lines (shown in red and blue, respectively, below), and you've fit a polynomial to those pixel positions. Next we'll compute the radius of curvature of the fit.



```python
def measure_curvature_pixel():
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    #print(left_curverad, right_curverad)
    return
```


```python
def measure_curvature_meter(ploty, lefty, leftx, righty, rightx):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    y_eval = np.max(ploty)
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curveradm = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curveradm = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Now our radius of curvature is in meters
    #print(left_curveradm, 'm', right_curveradm, 'm')
    return left_curveradm, right_curveradm
```


```python
import numpy as np
import matplotlib.pyplot as plt
# Generate some fake data to represent lane-line pixels
ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
quadratic_coeff = 3e-4 # arbitrary quadratic coefficient
# For each y position generate random x position within +/-50 pix
# of the line base position in each case (x=200 for left, and x=900 for right)
leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                              for y in ploty])
rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                for y in ploty])

leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
rightx = rightx[::-1]  # Reverse to match top-to-bottom in y


# Fit a second order polynomial to pixel positions in each fake lane line
left_fit = np.polyfit(ploty, leftx, 2)
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fit = np.polyfit(ploty, rightx, 2)
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

# Plot up the fake data
mark_size = 3
plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
plt.xlim(0, 1280)
plt.ylim(0, 720)
plt.plot(left_fitx, ploty, color='green', linewidth=3)
plt.plot(right_fitx, ploty, color='green', linewidth=3)
plt.gca().invert_yaxis() # to visualize as we do the images
plt.show()
```


![png](output_22_0.png)


## Tracking

After you've tuned your pipeline on test images, you'll run on a video stream, just like in the first project. In this case, however, you're going to keep track of things like where your last several detections of the lane lines were and what the curvature was, so you can properly treat new detections. To do this, it's useful to define a Line() class to keep track of all the interesting parameters you measure from frame to frame. Here's an example:


```python
# Define a class to receive the characteristics of each line detection
class OverlayLine():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        self.overlay = []
        
```


```python
global good_overlay
good_overlay = OverlayLine()
print (good_overlay.detected)
```

    False


## Sanity Check

Ok, so your algorithm found some lines. Before moving on, you should check that the detection makes sense. To confirm that your detected lane lines are real, you might consider:

    Checking that they have similar curvature
    Checking that they are separated by approximately the right distance horizontally
    Checking that they are roughly parallel


## Smoothing

Even when everything is working, your line detections will jump around from frame to frame a bit and it can be preferable to smooth over the last n frames of video to obtain a cleaner result. Each time you get a new high-confidence measurement, you can append it to the list of recent measurements and then take an average over n past measurements to obtain the lane position you want to draw onto the image.

## Drawing

Once you have a good measurement of the line positions in warped space, it's time to project your measurement back down onto the road! Let's suppose, as in the previous example, you have a warped binary image called warped, and you have fit the lines with a polynomial and have arrays called ploty, left_fitx and right_fitx, which represent the x and y pixel values of the lines. You can then project those lines onto the original image as follows:


```python
def draw_lanes(undist, binary_warped, ploty, left_fitx, right_fitx, dest, src):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Given src and dst points, calculate the perspective transform matrix
    Minv = cv2.getPerspectiveTransform(dest, src)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    
    #plt.imshow(final_result)
    return newwarp
```


```python
imgpoints, objpoints = find_chessboard_corners()
cimage = cv2.imread('calibration_test1.jpg')
mtx, dist = calibrateCamera(cimage, objpoints, imgpoints)
```

    ./camera_cal/calibration1.jpg False
    ./camera_cal/calibration10.jpg True
    ./camera_cal/calibration11.jpg True
    ./camera_cal/calibration12.jpg True
    ./camera_cal/calibration13.jpg True
    ./camera_cal/calibration14.jpg True
    ./camera_cal/calibration15.jpg True
    ./camera_cal/calibration16.jpg True
    ./camera_cal/calibration17.jpg True
    ./camera_cal/calibration18.jpg True
    ./camera_cal/calibration19.jpg True
    ./camera_cal/calibration2.jpg True
    ./camera_cal/calibration20.jpg True
    ./camera_cal/calibration3.jpg True
    ./camera_cal/calibration4.jpg False
    ./camera_cal/calibration5.jpg False
    ./camera_cal/calibration6.jpg True
    ./camera_cal/calibration7.jpg True
    ./camera_cal/calibration8.jpg True
    ./camera_cal/calibration9.jpg True



![png](output_30_1.png)



```python
#image = mpimg.imread('./test_images/test6.jpg')
src = np.float32([(195,720), (580,460), (705,460), (1130,720)])
dest = np.float32([(320,720), (320,0), (960,0), (960,720)])

def adv_lane_detect(image):
    undist, binary_warped = binary_threshold_undistort_warp(image, mtx, dist, src, dest)
    histogram = plot_hist(binary_warped)
    #plt.plot(histogram)
    #plt.show()
    #print(max(histogram))
    out_img,left_fit, right_fit, lefty, leftx, righty, rightx = slide_window_fit_polynomial(binary_warped)
    ploty, left_fitx, right_fitx = plot_sliding_window(binary_warped,out_img, left_fit, right_fit, lefty, leftx, righty, rightx, )
    left_curveradm, right_curveradm = measure_curvature_meter(ploty, lefty,leftx,righty, rightx)
    #print ('Radius of curvature: Left:',left_curveradm,'m, Right:',right_curveradm,'m')
    newwarp = draw_lanes(undist, binary_warped, ploty, left_fitx, right_fitx, dest, src)
    
    
    left_pos = left_fitx[left_fitx.size-1]
    right_pos = right_fitx[right_fitx.size-1]
    #print (left_pos, right_pos)
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    road_width_pixel = right_pos - left_pos
    road_width_meter = road_width_pixel * xm_per_pix
    #print ('Road width:',road_width_meter,'m')
    vehicle_pos = xm_per_pix * ((left_pos + (right_pos-left_pos) / 2) - image.shape[1]/2)
    #print ('Vehicle position from center of lane:',vehicle_pos,'m')
    
    #Do Sanity Check and use last know good value:
    if(abs(vehicle_pos) < 0.6 and left_curveradm > 300):
        good_overlay.overlay = newwarp
        good_overlay.line_base_pos = vehicle_pos
        good_overlay.radius_of_curvature = left_curveradm
        
    rad = "Radius: " + "%.2f" % round(good_overlay.radius_of_curvature,2) + "m"
    pos = "Position: "+ "%.2f" % round(good_overlay.line_base_pos,2) + "m"

    # Combine the result with the original image
    final_result = cv2.addWeighted(undist, 1, good_overlay.overlay, 0.3, 0)

    
    #print (rad, pos)
    # Combine the result with the original image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(final_result,rad,(100,100), font,2, (255,0,0), 5)
    cv2.putText(final_result,pos,(100,200), font,2, (255,0,0), 5)
    
    return final_result

images = glob.glob('./test_images/*.jpg')
for fname in images:
    image = mpimg.imread(fname)
    #print(fname)
    final_result = adv_lane_detect(image)
    plt.imshow(final_result)
    plt.show()
```


![png](output_31_0.png)



![png](output_31_1.png)



![png](output_31_2.png)



![png](output_31_3.png)



![png](output_31_4.png)



![png](output_31_5.png)



![png](output_31_6.png)



![png](output_31_7.png)



```python
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image_frame(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    result = adv_lane_detect(image)
    return result
```


```python
white_output = 'project_video_out_god.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image_frame) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)
```

    [MoviePy] >>>> Building video project_video_out_god.mp4
    [MoviePy] Writing video project_video_out_god.mp4


    100%|█████████▉| 1260/1261 [15:33<00:00,  1.48it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: project_video_out_god.mp4 
    
    CPU times: user 8min 39s, sys: 44.5 s, total: 9min 24s
    Wall time: 15min 42s

