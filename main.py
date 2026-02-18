from utils.video_utils import read_video, save_video
from trackers import Tracker

def main():
    #read_video
    video_frames = read_video('input_videos/football_analysis.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_track(video_frames)


    #save_video
    save_video(video_frames, 'output_videos/football_analysis_output')

if __name__ == '__main__':
    main()