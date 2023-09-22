import cv2
import numpy as np
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")

videoPath = 'labeled/4.hevc'
vid = cv2.VideoCapture(videoPath)
gt = np.loadtxt('labeled/4.txt')


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

pitch_yaw_pairs = []

i = 0
while True:

    cont, frame = vid.read()
    if not cont:
        break
    
    print(i)
    true_pitch, true_yaw = gt[i]
    i += 1

    # cv2.imshow('Edges', frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # bilateral_filtered_image = cv2.bilateralFilter(frame, 9, 75, 75)

    # Convert the cropped image to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Normalize the image
    normalized_image = gray_frame / 255.0

    # Adjust the contrast
    contrast_factor = 2.0  # You can adjust this value to increase or decrease contrast
    adjusted_image = (normalized_image - 0.5) * contrast_factor + 0.5

    # Clip pixel values to the range [0, 1]
    adjusted_image = np.clip(adjusted_image, 0, 1)

    # Convert back to 8-bit if needed
    adjusted_image = (adjusted_image * 255).astype(np.uint8)

    # equalized_image = cv2.equalizeHist(gray_frame)

    edges = cv2.Canny(adjusted_image, threshold1=100, threshold2=100)  # Adjust thresholds as needed

    # Define the coordinates of the three vertices of the triangle
    # In this example, let's assume the image dimensions are (width, height)
    width, height = frame.shape[1], frame.shape[0]
    bottom_left = (0, height)       # Bottom-left corner
    bottom_right = (width, height)  # Bottom-right corner
    middle = (width//2, height//2)       # Middle point at the top
    middle_left = (0, height//2)
    middle_right = (width, height//2)


    # Create an empty mask of the same size as the image
    mask = np.zeros((height, width), dtype=np.uint8)

    # Define the triangle by filling it with white (255)
    cv2.fillConvexPoly(mask, np.array([bottom_left, bottom_right, middle_right, middle_left]), 255)

    # Bitwise AND the mask with the image to obtain the cropped region
    cropped_triangle = cv2.bitwise_and(edges, edges, mask=mask)

    lines = cv2.HoughLinesP(
        cropped_triangle,
        rho=6,
        theta=np.pi / 60,
        threshold=50,
        lines=np.array([]),
        minLineLength=30,
        maxLineGap=25
    )

    left_x = []
    left_y = []
    right_x = []
    right_y = []
    left = []
    right = []

    if lines.any() != None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2:
                    continue 

                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.2:
                    continue
                
                bias = y1 - slope * x1
                center_x = frame.shape[1] // 2
                center_y = frame.shape[0] // 2
                line_y = slope * center_x + bias

                if abs(line_y - center_y) >= 100:
                    continue

                if slope < 0:
                    if x2 > width / 2 and x1 > width / 2:
                        continue 
                    left.append([x1, y1, x2, y2])
                    left_x += [x1, x2]
                    left_y += [y1, y2]
                else:
                    if x2 < width / 2 and x1 < width / 2:
                        continue 
                    right.append([x1, y1, x2, y2])
                    right_x += [x1, x2]
                    right_y += [y1, y2]

                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    vp_x = []
    vp_y = []

    for leftLine in left:
        for rightLine in right:
            x, y = find_intersection(leftLine, rightLine)
            vp_x.append(x)
            vp_y.append(y)
            # cv2.circle(frame, (int(x), int(y)), radius=2, color=(255, 0, 0), thickness=1)
    
    if (vp_x == []):
        pitch_yaw_pairs.append(pitch_yaw_pairs[-1])
        continue 

    vpx = int(np.mean(vp_x))
    vpy = int(np.mean(vp_y))
    yaw = np.arctan((np.mean(vp_x) - frame.shape[1]//2) / 910)
    pitch = -1 * np.arctan((np.mean(vp_y) - frame.shape[0]//2) / 910)

    # print(pitch, yaw)
    trueY = np.tan(-1 * true_pitch) * 910 + frame.shape[0]//2
    trueX = np.tan(true_yaw) * 910 + frame.shape[1]//2

    # cv2.circle(frame, (vpx, vpy), radius=2, color=(0, 255, 0), thickness=2)
    # cv2.circle(frame, (int(trueX), int(trueY)), radius=2, color=(0, 255, 255), thickness=2)

    # Combine x and y coordinates into a single array
    data = np.array(list(zip(vp_x, vp_y)))

    if len(data) < 4:
        pitch_yaw_pairs.append([pitch, yaw])
        continue

    # Specify the number of clusters (K)
    K = 4  # You can adjust this value

    # Apply K-means clustering to the combined data
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(data)

    # Get cluster centers
    cluster_centers = kmeans.cluster_centers_

    threshold_radius = 10.0  
    # Calculate the density of each cluster
    cluster_densities = []
    for center in cluster_centers:
        # Calculate the number of points within the threshold radius
        num_points_in_cluster = np.sum(np.linalg.norm(data - center, axis=1) <= threshold_radius)
        cluster_densities.append(num_points_in_cluster)

    most_dense_cluster_index = np.argmax(cluster_densities)
    x_final, y_final = cluster_centers[most_dense_cluster_index]
    
    cv2.circle(frame, (int(x_final), int(y_final)), radius=2, color=(0, 255, 0), thickness=2)

    yaw = np.arctan((x_final - frame.shape[1]//2) / 910)
    pitch = -1 * np.arctan((y_final - frame.shape[0]//2) / 910)

    pitch_yaw_pairs.append((pitch, yaw))

    # Display the cropped triangular region
    # cv2.imshow('Edges', frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

pitch_sum = 0
yaw_sum = 0

for i in range(len(pitch_yaw_pairs)):
    p, y = pitch_yaw_pairs[i]
    pitch_sum += p
    yaw_sum += y
    pitch_yaw_pairs[i] = [pitch_sum / (i + 1), yaw_sum / (i + 1)]


with open('test/4.txt', 'w') as file:
    # Iterate through the list of pitch and yaw pairs and write them to the file
    for pitch, yaw in pitch_yaw_pairs:
        print(f"{pitch} {yaw}", file=file)


        

    