import argparse
import logging
import sys
import whisper

import utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Parse input arguments
parser = argparse.ArgumentParser(description='Whisper Transcription',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path_video', type=str, default='video.mp4', help='Path to video file')
parser.add_argument('--path_audio', type=str, default='', help='Path to audio file')
parser.add_argument('--path_transcript', type=str, default='transcriptions/transcript.txt', help='Path to transcript file')
parser.add_argument('--resume', type=bool, default=False, help='Resume with AI')
parser.add_argument("--model", help="Indicate the Whisper model to download", default="small")
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


def main():
    if not args.path_video:
        logging.error("Please specify a video file")
        logging.info("Trying load audio file")
        if not args.path_audio:
            logging.error("Please specify a audio file")
            sys.exit()

    logging.info("Downloading Whisper model")
    model = whisper.load_model(args.model)

    logging.info("Loading video...")
    path = args.path_video if args.path_video else args.path_audio

    if args.path_video:
        logging.info("Get only the audio from the video")
        path = utils.convert_video_to_audio_path(path, "./audios/audio.mp3")

    logging.info("Transcribe the audio")
    result = model.transcribe(path)

    logging.info("Saving transcript")
    with open(args.path_transcript, 'w') as f:
        f.write(result)
        f.close()


if __name__ == '__main__':
    # Create a new instance of the app
    main()
