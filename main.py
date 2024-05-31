from utils import read_video, save_video
from trackers import Tracker, ball_track_id
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator
import numpy as np

def main():
    video_frames = read_video("input_videos/08fd33_4.mp4")

    # print the tracker
    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_tacks(video_frames, read_from_stub=True, stub_path="stubs/track.pkl")

    # Get object positions
    print("Adding position to tracks...")
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    print("Getting camera movement...")
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=True, stub_path="stubs/camera_movement_stub.pkl")

    print("Adjusting positions to tracks...")
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View Transformer
    print("Adding transformed position to tracks...")
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    print("Interpolating ball positions...")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    print("Adding speed and distance to tracks...")
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    print("Assigning team color...")
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["player"][0])

    for frame_num, player_track_dict in enumerate(tracks["player"]):
        for player_id, track in player_track_dict.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num],
                track["bbox"],
                player_id)
            tracks["player"][frame_num][player_id]["team"] = team
            tracks["player"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]

    # Assign Ball Aquisition
    print("Assigning player ball aquisition...")
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, players_track in enumerate(tracks["player"]):
        ball_bbox = tracks["ball"][frame_num][ball_track_id]["bbox"]
        assigned_player = player_assigner.assign_ball_to_player(players_track, ball_bbox)

        if assigned_player != -1:
            tracks["player"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(tracks["player"][frame_num][assigned_player]["team"])
        else:
            team_ball_control.append(team_ball_control[-1])

    team_ball_control = np.array(team_ball_control)

    # Draw object
    # Draw object ellipse
    print("Drawing object annotations...")
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Draw camera movement
    print("Drawing camera movement...")
    camera_movement_estimator.draw_camera_movement_in_place(output_video_frames, camera_movement_per_frame)

    # Draw speed and distance statistics
    print("Drawing speed and distance...")
    speed_and_distance_estimator.draw_speed_and_distance_in_place(output_video_frames, tracks)

    print("Saving video...")
    save_video(output_video_frames, "output_videos/output.avi")


if __name__ == "__main__":
    main()