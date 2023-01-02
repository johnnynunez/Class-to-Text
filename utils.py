from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.editor import VideoFileClip


def convert_video_to_audio(video_file):
    video = VideoFileClip(video_file)
    audio = video.audio
    return audio

def convert_video_to_audio_path(video_file, output_path):
    video = VideoFileClip(video_file)
    audio = video.audio
    audio.write_audiofile(output_path)
    return output_path


