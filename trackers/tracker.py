from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import os
import pickle
import sys
sys.path.append("../")
from utils import get_bbox_center, get_bbox_width

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        frame_len =  len(frames)
        batch_size = 20
        detections = []
        for i in range(0, frame_len, batch_size):
            detections += self.model.predict(frames[i:i+batch_size], conf=0.1)
        return detections

    def get_object_tacks(self, frames, read_from_stub=False, stub_path=None):
        detections = self.detect_frames(frames)

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks
            
        obj_classes = ["player", "referee", "ball"]
        
        tracks = {obj_class:[] for obj_class in obj_classes}

        for frame_num, detection in enumerate(detections):
            class_names = detection.names
            class_names_inv = {class_name:class_id for class_id, class_name in class_names.items()}

            # Convert to Supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert Goalkeeper to Player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if class_names[class_id] == "goalkeeper":
                    print(f"Replacing goalkeeper for player in object {object_ind}")

                    detection_supervision.class_id[object_ind] = class_names_inv["player"]

                    # As of supervision v 0.20.0, there's a dictionaary property, data, with the respective classes
                    detection_supervision.data["class_name"][object_ind] = "player"


            detection_with_tracking = self.tracker.update_with_detections(detection_supervision)

            for obj_class in obj_classes:
                tracks[obj_class].append({})

            # collect player and referee bbox in frame
            for tracked_object_in_frame in detection_with_tracking:
                bbox = tracked_object_in_frame[0].tolist()
                class_id = tracked_object_in_frame[3]
                track_id = tracked_object_in_frame[4]
                class_name = class_names[class_id]

                if class_name in ["player", "referee"]:
                    tracks[class_name][frame_num][track_id] = {"bbox": bbox}

            # collect ball bbox in frame
            ball_track_id = 1
            for detected_obj_in_frame in detection_supervision:
                bbox = detected_obj_in_frame[0].tolist()
                class_id = detected_obj_in_frame[3]

                if class_id == class_names_inv["ball"]:
                    tracks["ball"][frame_num][ball_track_id] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)
 
    def draw_eclipse(self, frame, bbox, color, track_id = None):
        y2 = int(bbox[3])
        width = get_bbox_width(bbox)
        x_center, _ = get_bbox_center(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        if track_id is not None:
            # draw rectangle
            rectangle_width = 40
            rectangle_height = 20

            x1_rect = x_center - rectangle_width//2
            x2_rect = x_center + rectangle_width//2
            y1_rect = y2 - rectangle_height//2 + 15
            y2_rect = y2 + rectangle_height//2 + 15

            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color=color,
                thickness=cv2.FILLED
            )

            # write text
            x_text = x1_rect + 12
            if int(track_id) > 99:
                x_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                org=(int(x_text), int(y1_rect + 15)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(0,0,0),
                thickness=2
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])    # using y1 because we want the apex on the ball
        x, _ = get_bbox_center(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)   # border, triangle as edges not filled

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["player"][frame_num]
            referee_dict = tracks["referee"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Draw player
            for track_id, obj in player_dict.items():
                self.draw_eclipse(frame, obj["bbox"], (0,0,255), track_id)

            # Draw referee
            for _, obj in referee_dict.items():
                self.draw_eclipse(frame, obj["bbox"], (0,255,255))

            # Draw ball
            for _, obj in ball_dict.items():
                self.draw_triangle(frame, obj["bbox"], (0,255,0))

            output_video_frames.append(frame)

        return output_video_frames
