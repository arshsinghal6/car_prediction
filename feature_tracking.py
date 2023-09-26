import cv2
import numpy as np

videoPath = 'labeled/0.hevc'
vid = cv2.VideoCapture(videoPath)

max_features = 1000
detector = cv2.ORB_create(max_features)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

prev_frame, prev_keypoints, prev_desc = None, None, None
curr_frame, curr_keypoints, curr_desc = None, None, None

intrinsic_matrix = np.array([[910., 0, 1164/2],
                            [0, 910., 874/2],
                            [0, 0, 1]])

def find_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Calculate the slopes of the lines
    slope1 = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else float('inf')
    slope2 = (y4 - y3) / (x4 - x3) if x4 - x3 != 0 else float('inf')

    # Check if the lines are parallel
    if slope1 == slope2:
        return None  # Lines are parallel and don't intersect

    # Calculate the intersection point
    if slope1 == float('inf'):
        x = x1
        y = slope2 * (x - x3) + y3
    elif slope2 == float('inf'):
        x = x3
        y = slope1 * (x - x1) + y1
    else:
        x = (y1 - y3 + slope2 * x3 - slope1 * x1) / (slope2 - slope1)
        y = slope1 * (x - x1) + y1

    return x, y

j = 0
while j < 10:
    j += 1
    print(j)
    cont, frame = vid.read()
    if not cont:
        break


    mask = np.zeros_like(frame, dtype=np.uint8)

    # Define the region where you want to detect keypoints and compute descriptors
    # In this example, I'm creating a circular mask in the center of the image
    center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
    radius = 10
    cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), -1)

    # print(mask)

    prev_frame, prev_keypoints, prev_desc = curr_frame, curr_keypoints, curr_desc

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    curr_frame = frame_gray
    curr_keypoints, curr_desc = detector.detectAndCompute(frame_gray, None)

    if j <= 1:
        continue

    matches = matcher.match(prev_desc, curr_desc)

    # Extract matched keypoints
    src_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([curr_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    lines = []
    for i in range(len(src_pts)):
        xs, ys = src_pts[i][0][0], src_pts[i][0][1]
        xd, yd = dst_pts[i][0][0], dst_pts[i][0][1]
        dist = np.sqrt((xs - xd)**2 + (ys - yd)**2)
        if dist <= 15:
            continue

        slope1 = (yd - ys) / (xd - xs)
        if abs(slope1) < 0.2:
            continue
        cv2.line(frame, (int(xs), int(ys)), (int(xd), int(yd)), (0, 0, 255), 2)
        lines.append([xs, ys, xd, yd])
    
    vp_x = []
    vp_y = []
    for line1 in lines:
        for line2 in lines:
            if line1 == line2:
                continue
            inter = find_intersection(line1, line2)
            if inter:
                vp_x.append(inter[0])
                vp_y.append(inter[1])
    
    vpx = int(np.mean(vp_x))
    vpy = int(np.mean(vp_y))
    # print(vpx, vpy)
    cv2.circle(frame, (vpx, vpy), radius=2, color=(255, 0, 0), thickness=3)

    # for pt in src_pts:
    #     x, y = pt[0][0], pt[0][1]
    #     cv2.drawMarker(frame, (int(x), int(y)), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3, line_type=cv2.LINE_8)

    # for pt in dst_pts:
    #     x, y = pt[0][0], pt[0][1]
    #     cv2.drawMarker(frame, (int(x), int(y)), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3, line_type=cv2.LINE_8)

    cv2.imshow('Frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

        

    