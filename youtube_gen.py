import openai
import tempfile
import numpy as np
import pandas as pd
from pytube import YouTube
import os
import yt_dlp
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

video_dict = {
    "url": [],
    "title": [],
    "content": []
}

def download(video_id: str) -> str:
    video_url = f'https://www.youtube.com/watch?v={video_id}'
    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'paths': {'home': 'audio/'},
        'outtmpl': {'default': '%(id)s.%(ext)s'},
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download([video_url])
        if error_code != 0:
            raise Exception('Failed to download video')

    return f'audio/{video_id}.m4a'

def video_to_audio(video_URL):
    # Get the video
    video = YouTube(video_URL)
    video_dict["url"].append(video_URL)
    try:
        video_dict["title"].append(video.title)
    except:
        video_dict["title"].append("Title not found")


    # Convert video to Audio
    #  streams = youtube_video.streams.get_audio_only().download(filename=audio_file_path)
    temp_path = "audio.mp4"
    audio = video.streams.get_audio_only()
    
    # audio.download(filename=temp_path)

    # temp_dir = tempfile.mkdtemp()
    # variable = np.random.randint(1111, 1111111)
    # file_name = f'recording{variable}.mp3'

    # Save to destination
    output = audio.download(output_path=temp_path)

    audio_file = open(output, "rb")
    textt = openai.Audio.translate("whisper-1", audio_file)["text"]

    return textt



def get_url_text(url_link):
    transcription = video_to_audio(url_link)
    video_dict["content"].append(transcription)
    return video_dict["content"]

