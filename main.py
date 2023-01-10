import argparse
import logging
import os
import sys

import openai
import whisper
from tqdm import tqdm

import utils

CHUNK_SIZE = 4096

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(ROOT_DIR, './keys/openai_key.txt'), 'r') as f:
    key = f.readline().strip()
    openai.api_key = key

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
parser.add_argument('--path_transcript', type=str, default='transcriptions/transcript.txt',
                    help='Path to transcript file')
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
    # Load model
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
    result = model.transcribe(path, fp16=args.fp16, verbose=False)
    # traduce result

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
        # split text in chunks
        chunks = utils.reduce_prompt(transcript, CHUNK_SIZE)
        # # Split the text into chunks
        # text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        # send chunks to AI
        # Initialize an empty list to store the generated responses
        responses = []
        for chunk in tqdm(chunks):
            response = openai.Completion.create(
                engine="davinci",
                prompt=chunk,  # f"{chunk}",
                temperature=0.9,
                max_tokens=150,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0.6,
                stop=["\n"]
            )
            # save response in new paragraph
            responses.append(response['choices'][0]['text'] + "\n")
        final_output = "".join(responses)
        with open('./resume/resume.txt', 'w') as f:
            f.write(final_output)
            f.close()
        logging.info("Saving Resume")


if __name__ == '__main__':
    # Create a new instance of the app
    main()
