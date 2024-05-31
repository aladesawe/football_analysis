import cv2
import numpy as np
import pickle
import os
import sys
sys.path.append("../")
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self, frame):

        # so camera movement is not so little/statistically significant
        self.minimum_camera_movement = 5

        # optical flow params
        self.lk_params = dict(
            winSize = (15,15),    # Search size
            maxLevel = 2,   # pyramids to downscale the image to get larger features/downscale up to twice
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # stopping criteria or loop 10 times and find no score above 0.03
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:,0:20] = 1   # first rows of pixels/banner from top
        mask_features[:,900:1050] = 1   # last rows of pixels/banner from bottom

        # review documentation for cv2.goodFeatureToTrack
        self.features = dict(
            maxCorners = 100,   # max number of corners we can utilize for the features
            qualityLevel = 0.3, # the higher, the better the features but the lesser the number of features
            minDistance = 3,    # btw pixels
            blockSize = 7,  # Search size of the features
            mask = mask_features    # where to extract the features from
        )

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info["position"]
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]["position_adjusted"] = position_adjusted

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # Read the stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        camera_movement = [[0,0]]*len(frames)
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1,len(frames)):
            new_gray = cv2.cvtColor(frames[frame_num],cv2.COLOR_BGR2GRAY)
            new_features, _,_ = cv2.calcOpticalFlowPyrLK(old_gray,new_gray,old_features,None,**self.lk_params)

            # measure distance between old and new features, to see if there's movement in the camera movement
            # each frame is going to have multiple features, so getting max distance between any 2
            max_distance = 0
            camera_movement_x, camera_movement_y = 0,0

            for i, (new,old) in enumerate(zip(new_features,old_features)):
                new_feature_point = new.ravel()
                old_feature_point = old.ravel()

                distance = measure_distance(new_feature_point, old_feature_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_feature_point, new_feature_point)

            if max_distance > self.minimum_camera_movement:
                camera_movement[frame_num]= [camera_movement_x,camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(new_gray, **self.features)

            old_gray = new_gray.copy()

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def draw_camera_movement_in_place(self, frames, camera_movement_per_frame):
        for frame_num, frame in enumerate(frames):
            overlay = frame.copy()
            cv2.rectangle(overlay,(0,0),(500,100),(255,255,255),-1) # filled cv2.FILLED == -1
            alpha = 0.6 # transparency
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame, f"Camera Movement X:{x_movement:.2f}",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frame = cv2.putText(frame, f"Camera Movement Y:{y_movement:.2f}",(10,60), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
