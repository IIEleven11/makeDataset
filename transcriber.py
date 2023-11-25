import os
import csv
from phonemizer import phonemize
import speech_recognition as sr
from CloneXTTSv2.vits.text import _clean_text 


def transcribe_audio_files_in_directory(
    directory_path, txt_file_path, bad_audio_directory
):
    r = sr.Recognizer()

    # Create bad audio directory
    os.makedirs(bad_audio_directory, exist_ok=True)

    with open(txt_file_path, "w", newline="") as txtfile:
        filenames = sorted(os.listdir(directory_path))

        for filename in filenames:
            if filename.endswith(".wav"):
                file_path = os.path.join(directory_path, filename)

                with sr.AudioFile(file_path) as source:
                    audio = r.record(source)
                    try:
                        # transcribe
                        transcription = r.recognize_google(audio)
                        # clean the transcription
                        cleaned_transcription = _clean_text(
                            transcription, ["english_cleaners2"]
                        )
                        # phonemize
                        phonemized_transcription = phonemize(
                            cleaned_transcription, language="en-us"
                        )
                        txtfile.write(f"{filename}|{phonemized_transcription}|1\n")
                    except sr.UnknownValueError:
                        print(
                            f"Google Web Speech API could not understand audio for file: {filename}. Moving it to badAudio directory."
                        )
                        # move badAudio directory
                        os.rename(
                            file_path, os.path.join(bad_audio_directory, filename)
                        )
                    except sr.RequestError as e:
                        print(
                            "Could not request results from Google Web Speech API service; {0}".format(
                                e
                            )
                        )


transcribe_audio_files_in_directory(
    "/home/eleven/segmenter/output", # path to all wav files
    "/home/eleven/segmenter/txtfile/train_data.txt", # path to where you want the txt to go
    "/home/eleven/segmenter/badaudio/badaudio", # path to put the bad wav files
)
