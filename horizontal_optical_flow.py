import cv2
import numpy as np

class flow_detector:

    def __init__(self):

        self.prev_frame, self.prev_keypoints = None, None
        self.curr_frame, self.curr_keypoints = None, None
        self.past_frames = []
        self.past_vectors = [[0, 0]]

        self.intrinsic_matrix = np.array([[910., 0, 1164/2],
                                    [0, 910., 874/2],
                                    [0, 0, 1]])

        self.feature_params = dict( maxCorners = 300,
                            qualityLevel = 0.2,
                            minDistance = 10,
                            blockSize = 10 )

        self.lk_params = dict( winSize  = (10,10), maxLevel = 1,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        width, height = 1164, 874
        bottom_left = (int(width * 1/3), int(height * 3/4))      
        bottom_right = (int(width * 2/3), int(height * 3/4))  
        top_left = (int(width * 1/3), 0)
        top_right = (int(width * 2/3), 0)
        
        self.mask = np.zeros((height, width), dtype=np.uint8)

        cv2.fillConvexPoly(self.mask, np.array([bottom_left, bottom_right, top_right, top_left]), 255)

        self.j = 0

    def flow_frame(self, frame):
        self.j += 1

        if self.curr_frame is not None and self.curr_frame.all() != None:
            self.prev_frame = self.curr_frame.copy()
        
        if self.prev_frame is not None and self.prev_frame.all() != None:
            cropped_frame = cv2.bitwise_and(self.prev_frame, self.prev_frame, mask=self.mask)
            self.prev_keypoints = cv2.goodFeaturesToTrack(cropped_frame, mask = None, **self.feature_params)  
            self.past_frames.append([self.prev_frame, self.prev_keypoints])


        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.curr_frame = frame_gray


        if self.j <= 1:
            return 0, 0


        past_frame, p0 = self.past_frames.pop(0)

        cropped_prev_frame = cv2.bitwise_and(past_frame, past_frame, mask=self.mask)
        cropped_curr_frame = cv2.bitwise_and(self.curr_frame, self.curr_frame, mask=self.mask)

        self.curr_keypoints, st, err = cv2.calcOpticalFlowPyrLK(cropped_prev_frame, cropped_curr_frame, p0, None, **self.lk_params)

        good_new = self.curr_keypoints[st==1]
        good_old = p0[st==1]

        mid_x = frame.shape[1] // 2
        mid_y = frame.shape[0] // 2
        # frame = cv2.circle(frame, (int(mid_x),int(mid_y)), 2, (255, 0, 0))
        new_points = []

        for (new, old) in zip(good_new,good_old):
            a,b = new.ravel()
            c,d = old.ravel()
            
            delta_x = a - c
            delta_y = b - d
            dist = np.sqrt(delta_x ** 2 + delta_y ** 2)
            if dist > 75:
                continue

            if abs(delta_y) > 10 or abs(delta_x) > 30:
                continue

            new_points.append([delta_x, delta_y])

            # frame = cv2.circle(frame, (int(a),int(b)), 2, (255, 255, 0))
            # frame = cv2.line(frame, (int(a),int(b)),(int(c),int(d)), (0, 255, 0), 2)
        
        motion_vector = [0, 0]
        if new_points != []:
            motion_vector = np.mean(np.array(new_points), axis=0)

        self.past_vectors.append(motion_vector)

        stdev_away = 0
        avg_vector = None
        if len(self.past_vectors) >= 75:
            avg_vector = np.mean(self.past_vectors[-75:], axis=0)
            avg_x = np.mean(self.past_vectors, axis=0)[0]
            stdev = np.std(self.past_vectors, axis=0)[0]
            curr_deltaX = avg_vector[0]
            stdev_away = abs((curr_deltaX - avg_x) / stdev)
            # print(stdev_away)
  
    
        if avg_vector is not None and not np.isnan(avg_vector[0]):
            # frame = cv2.line(frame, (int(mid_x),int(mid_y)),(int(avg_vector[0]),int(mid_y)), (255, 0, 0), 2)
            pass
        else:
            return 0, 0

        return [-1 * avg_vector[0], stdev_away]



        

    