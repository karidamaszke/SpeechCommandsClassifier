import math
import struct
import time

import numpy as np
import pyaudio
from pydub import AudioSegment
from pydub.silence import split_on_silence

TIMEOUT_LENGTH = 2
SHORT_NORMALIZE = (1.0 / 32768.0)


class AudioRecorder:
    def __init__(self):
        self.__p = pyaudio.PyAudio()
        self.__threshold = 30
        self.__chunk_size = 1024
        self.__segment = None
        self.__stream = self.__p.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=16000,
                                      input=True,
                                      output=True,
                                      frames_per_buffer=self.__chunk_size)

    def listen(self):
        print('Listening start...')
        while True:
            chunk = self.__stream.read(self.__chunk_size)
            if self.__rms(chunk) > self.__threshold:
                self.__record()
                break
        print("Listening stop...")

    def get_words(self) -> [np.array]:
        if self.__segment is not None:
            words = []
            audio_chunks = split_on_silence(self.__segment,
                                            min_silence_len=300,
                                            silence_thresh=-50)

            for i, chunk in enumerate(audio_chunks):
                chunk = self.__pad_chunk_to_1s(chunk)
                samples = np.array(chunk.get_array_of_samples())
                samples = self.__pad_array_to_16k(samples)
                words.append(samples)

            return words

    def __record(self):
        print('Voice detected...')
        rec = []
        current = time.time()
        end = time.time() + TIMEOUT_LENGTH
        while current <= end:
            data = self.__stream.read(self.__chunk_size)
            if self.__rms(data) >= self.__threshold:
                end = time.time() + TIMEOUT_LENGTH
            current = time.time()
            rec.append(data)

        self.__segment = AudioSegment(data=b''.join(rec), sample_width=2, frame_rate=16000, channels=1)

    @staticmethod
    def __rms(frame):
        count = len(frame) / 2
        shorts = struct.unpack("%dh" % count, frame)
        sum_squares = sum([(sample * SHORT_NORMALIZE) ** 2 for sample in shorts])
        rms = math.pow(sum_squares / count, 0.5)
        return rms * 1000

    @staticmethod
    def __pad_chunk_to_1s(chunk) -> AudioSegment:
        pad = 1000 - len(chunk)
        if pad > 0:
            left_pad = pad // 2
            right_pad = pad - left_pad
            return AudioSegment.silent(duration=left_pad) + chunk + AudioSegment.silent(duration=right_pad)
        else:
            pad = abs(pad // 2)
            return chunk[pad:1000 + pad]

    @staticmethod
    def __pad_array_to_16k(arr) -> np.array:
        pad = 16000 - len(arr)
        left_pad = pad // 2
        right_pad = pad - left_pad
        return np.pad(arr, (left_pad, right_pad), 'constant')
