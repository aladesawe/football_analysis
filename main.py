from utils import read_video, save_video
from trackers import Tracker

def main():
    video_frames = read_video("input_videos/08fd33_4.mp4")

    # print the tracker
    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_tacks(video_frames, read_from_stub=True, stub_path="stubs/track.pkl")

    # Draw object
    # Draw object ellipse
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    save_video(output_video_frames, "output_videos/output.avi")


if __name__ == "__main__":
    main()