# car_prediction
My take on the comma.ai calibration challenge

## The Approach
The goal of the challenge is to predict the pitch and yaw of the car's motion in the frame of the mounted dashcam. To obtain the pitch and yaw in the camera frame, we have to understand how 3D points from the camera coordinate system are projected onto a 2D pixel coordinate system. The camera coordinate system is defined as the z-axis pointing out from the camera, the x-axis as side-to-side, and the y-axis as up-down. Since we only have data from a single camera angle, the projection onto the pixel coordinate system means any depth information (Z-axis) is lost. However, the following formulas provide us with the X/Z and Y/Z ratio:

X/Z = 1/f * (u - c_x)
Y/Z = 1/f * (v - c_y),  where f = focal length, (u, v) = pixel coordinate, (c_x, c_y) = center of frame

Taking the arctan of these ratio provide us with the yaw and pitch, respectively. So, given the (u, v) coordinate of where the car is headed in the camera frame, we can compute the pitch and yaw of the car. To determine this coordinate, I make the assumption that the car is headed towards the vanishing point in the frame. 

## Computing Vanishing Point
I experimented with a few methods to compute the vanishing point in the frame, but I had the most success with trying to detect road lines and finding their intersection. The CV pipeline I created included the following stages:
  * Convert to grayscale
  * Increase contrast
  * Detect edges using Canny
  * Mask image to only look at bottom half (approximating where the road is)
  * Hough Line Transform
  * Detect left and right lines based on slope
  * Filter out extreme lines (too flat, don't intersect close to center, etc.)
  * Find intersection between all right and left lines
  * Use K-means to detect cluster of intersection points with highest density
  * Use a windowed moving average to get vanishing point using center of cluster as a new estimate
  * Convert VP to pitch, yaw

After expermienting a lot, this is the pipeline I was most happy with. Though, I think it can get better with other techniques/heuristics. After testing with the labeled data by Comma, the pipeline did really well on clear, straight highways (able to get ~0.5% error). The program did a little worse on videos of inner streets (~3-5% error). However, one of the labeled videos included a sharp left turn. The VP technique breaks down in such scenarios, bringing my total error on the labeled dataset to (~11% error). Regardless, I was excited to get a decent error rate using a pure CV technique. 

## Horizontal Optical Flow
Not accounting for turning felt like cheating, so I tried fixing it. I went back to a previous method I had tried for VP: optical flow. Since a turn doesn't really affect the pitch, I was only interested in the change in yaw, or horizontal flow. The pipeline was roughly:
  * Mask previous frame
  * Find good features to track using corner detection
  * Convert current frame to gray scale
  * Mask current frame
  * Calculate optical flow between two frames using Lucas-Kanade method
  * Filter out some noise in matched features
  * Take the mean of all deltas between matched features to get motion vector
  * Compute moving average for the mean delta vector
  * Bring delta_x to 0 if within a standard deviation of past data
  * Use delta_x to compute new VP

This method had varying success for detecting turning motion. While it was able to detect sharp turns in the video successfully, there were also false-positives that would influence the VP when it shouldn't due to there being so much noise. Also, the lack of turns in the labeled data made it more challenging to experiment with the calculations without guessing. I was able to bring the error on the labeled dataset down to 5%, but I'm not sure about how successful it will be on the unlabeled data. There are a lot more parameters (threshold for corner detection, region of interest, number of corners, etc.) that I could mess around with if I had more time. For now, let's just say this portion will stay a WIP. 

## Artifacts
I have created some videos to visualize my predictions under the prediction_videos folder. There are a lot of messy visualizations mainly because I used them for testing. Generally, the red lines in the video indicate where a line on the road is detected. Tiny blue dots are the clusters of intersection points between left and right lines. The green dot is my VP prediction and the yellow dot (for the labeled dataset videos) is the ground truth. Videos labelled "optical" show some of the keypoint movements from frame to frame, but there is a lot of noise. Videos labeled with "_turn" are ones that combine the VP detection and horizontal flow to try to account for turning. 

Overall, I had a lot of fun on the project and I hope I can come back to it one day to refine some of it. Thanks to the team at comma.ai for the challenge!



