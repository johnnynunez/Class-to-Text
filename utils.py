import math

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


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def reduce_prompt(prompt, max_tokens=4097):
    tokens = prompt.split()
    if len(tokens) <= max_tokens:
        return [prompt]
    else:
        n = math.ceil(len(tokens) / max_tokens)
        return list(chunks(tokens, n))
