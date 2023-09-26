import cv2
import numpy as np
from sklearn.cluster import KMeans
import horizontal_optical_flow as flow
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

for num in [5, 6, 7, 8, 9]:
    videoPath = f'unlabeled/{num}.hevc'
    vid = cv2.VideoCapture(videoPath)
    # gt = np.loadtxt('labeled/4.txt')
    out = cv2.VideoWriter(f'prediction_videos/pred{num}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(vid.get(3)), int(vid.get(4))))
    pitch_yaw_pairs = []
    sum_x = []
    sum_y = []
    display_x = int(1164/2)
    display_y = int(874/2)
    TURN_SCALE = 20
    STDEV_THRESHOLD_LOW = 1
    STDEV_THRESHOLD_HIGH = 1.8
    flowDetector = flow.flow_detector()

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

    def moving_average(x, y):
        # if len(sum_x) == 300:
        #     sum_x.pop(0)
        #     sum_y.pop(0)

        sum_x.append(x)
        sum_y.append(y)

        avg_x = np.average(sum_x)
        avg_y = np.average(sum_y)

        return int(avg_x), int(avg_y)

    graph_hor = []

    i = 0
    while True:

        cont, frame = vid.read()
        if not cont:
            # xs = np.linspace(0, 40, i)
            # plt.plot(xs, graph_hor, label="horizontal_movement", color='blue')
            # plt.legend()

            # # Show the plot
            # plt.grid(True)
            # plt.show()
            break

        # print(i)
        # true_pitch, true_yaw = gt[i]
        i += 1

        horizontal_movement, stdev_away = flowDetector.flow_frame(frame)
        horizontal_movement *= TURN_SCALE
        horizontal_movement = 0
        if stdev_away < STDEV_THRESHOLD_LOW or stdev_away > STDEV_THRESHOLD_HIGH:
            horizontal_movement = 0
        
        print(i, horizontal_movement)
        graph_hor.append(horizontal_movement)

        # Convert the cropped image to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Normalize the image
        normalized_image = gray_frame / 255.0

        # Adjust contrast
        contrast_factor = 2.0  
        adjusted_image = (normalized_image - 0.5) * contrast_factor + 0.5

        # Clip pixel values to the range [0, 1]
        adjusted_image = np.clip(adjusted_image, 0, 1)
        adjusted_image = (adjusted_image * 255).astype(np.uint8)

        # equalized_image = cv2.equalizeHist(gray_frame)

        edges = cv2.Canny(adjusted_image, threshold1=100, threshold2=100)  


        width, height = frame.shape[1], frame.shape[0]
        bottom_left = (0, int(height * 3/4))      
        bottom_right = (width, int(height * 3/4))  
        middle = (width//2, height//2)       
        middle_left = (0, int(height * 0.6))
        middle_right = (width, int(height * 0.6))
        top_right = (int(width * 0.70), int(height * 0.55))
        top_left = (int(width * 0.25), int(height * 0.55))


        # Crop image
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.array([bottom_left, bottom_right, middle_right, top_right, top_left, middle_left]), 255)
        cropped_image = cv2.bitwise_and(edges, edges, mask=mask)


        lines = cv2.HoughLinesP(
            cropped_image,
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

        # Find intersection points between lines detected on the right and left side of frame
        if lines is not None and lines.any() != None:
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
                cv2.circle(frame, (int(x), int(y)), radius=2, color=(255, 0, 0), thickness=1)
        
        # trueY = np.tan(-1 * true_pitch) * 910 + frame.shape[0]//2
        # trueX = np.tan(true_yaw) * 910 + frame.shape[1]//2

        if (vp_x == []):
            # display_x, display_y = moving_average(display_x, display_y)

            display_x += horizontal_movement

            cv2.circle(frame, (int(display_x), int(display_y)), radius=2, color=(0, 255, 0), thickness=2)

            # if not np.isnan(trueX) and not np.isnan(trueY):
            #     cv2.circle(frame, (int(trueX), int(trueY)), radius=2, color=(0, 255, 255), thickness=2)

            yaw = np.arctan((display_x - frame.shape[1]//2) / 910)
            pitch = -1 * np.arctan((display_y - frame.shape[0]//2) / 910)

            pitch_yaw_pairs.append((pitch, yaw))
            
            out.write(frame)
            continue 

        vpx = int(np.mean(vp_x))
        vpy = int(np.mean(vp_y))


        # cv2.circle(frame, (vpx, vpy), radius=2, color=(0, 255, 0), thickness=2)

        # Combine x and y coordinates into a single array
        data = np.array(list(zip(vp_x, vp_y)))

        if len(data) < 4:
        # if True:        
            display_x, display_y = moving_average(vpx, vpy)

            display_x += horizontal_movement

            yaw = np.arctan((display_x - frame.shape[1]//2) / 910)
            pitch = -1 * np.arctan((display_y - frame.shape[0]//2) / 910)

            pitch_yaw_pairs.append([pitch, yaw])

            cv2.circle(frame, (int(display_x), int(display_y)), radius=2, color=(0, 255, 0), thickness=2)

            # if not np.isnan(trueX) and not np.isnan(trueY):
            #     cv2.circle(frame, (int(trueX), int(trueY)), radius=2, color=(0, 255, 255), thickness=2)

            # frame = cv2.bitwise_and(frame, frame, mask=mask)
            out.write(frame)

            continue
        
        # Number of clusters
        K = 4 

        # Apply K-means clustering to the combined data
        kmeans = KMeans(n_clusters=K)
        kmeans.fit(data)
        cluster_centers = kmeans.cluster_centers_

        threshold_radius = 10.0  

        # Calculate the density of each cluster
        cluster_densities = []
        for center in cluster_centers:
            num_points_in_cluster = np.sum(np.linalg.norm(data - center, axis=1) <= threshold_radius)
            cluster_densities.append(num_points_in_cluster)

        most_dense_cluster_index = np.argmax(cluster_densities)
        x_final, y_final = cluster_centers[most_dense_cluster_index]

        display_x, display_y = moving_average(x_final, y_final)

        display_x += horizontal_movement
        
        cv2.circle(frame, (int(display_x), int(display_y)), radius=2, color=(0, 255, 0), thickness=2)

        # if not np.isnan(trueX) and not np.isnan(trueY):
        #     cv2.circle(frame, (int(trueX), int(trueY)), radius=2, color=(0, 255, 255), thickness=2)

        # frame = cv2.bitwise_and(frame, frame, mask=mask)
        out.write(frame)

        yaw = np.arctan((display_x - frame.shape[1]//2) / 910)
        pitch = -1 * np.arctan((display_y - frame.shape[0]//2) / 910)


        pitch_yaw_pairs.append((pitch, yaw))

        # cv2.imshow('Edges', frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    with open(f'test/{num}.txt', 'w') as file:
        for pitch, yaw in pitch_yaw_pairs:
            print(f"{pitch} {yaw}", file=file)


        

    