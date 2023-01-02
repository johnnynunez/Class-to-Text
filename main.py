import argparse
import logging
import sys
import whisper
import openai
import utils

openai.api_key = 'YOUR_API_KEY'

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
parser.add_argument('--path_video', type=str, help='Path to video file')
parser.add_argument('--path_audio', type=str, default='', help='Path to audio file')
parser.add_argument('--path_transcript', type=str, default='transcriptions/transcript.txt', help='Path to transcript file')
parser.add_argument('--resume', type=bool, default=False, help='Resume with AI')
parser.add_argument("--model", help="Indicate the Whisper model to download", default="small")
parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
parser.add_argument('--fp16', type=bool, default=False, help='Use FP16')
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
            exit()

    logging.info("Downloading Whisper model")
    model = whisper.load_model(args.model, device=args.device)

    logging.info("Loading video...")
    path = args.path_video if args.path_video else args.path_audio

    if args.path_video:
        logging.info("Get only the audio from the video")
        path = utils.convert_video_to_audio_path(path, "./audios/audio.mp3")

    logging.info("Transcribe the audio")
    result = model.transcribe(path, fp16=args.fp16, verbose=True)

    logging.info("Saving transcript")
    # save text with line breaks
    with open(args.path_transcript, 'w') as f:
        for s in result["segments"]:
            start = s['start']
            end = s['end']
            text = s['text']
            f.write(text + "\n")
        f.close()

    logging.info("Transcript saved in {}".format(args.path_transcript))

    if args.resume:
        logging.info("Resume with AI")
        # load txt file
        transcript = open(args.path_transcript, 'r').read()
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=transcript,
            max_tokens=1024,
            temperature=0.5,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        logging.info("Saving Resume")
        with open('resume.txt', 'w') as f:
            f.write(response['choices'][0]['text'])
            f.close()




if __name__ == '__main__':
    # Create a new instance of the app
    main()
