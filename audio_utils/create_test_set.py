import os
import sys
import numpy as np
import sounddevice as sd
from time import sleep
from scipy.io.wavfile import write

from model_manager import LABELS


def print_double():
    print(50 * '=')


def get_word(waveform):
    try:
        start_index = np.where(waveform > 0.03)[0][0]
        stop_index = np.where(waveform > 0.03)[0][-1]
        waveform = waveform[int(start_index)-100:int(stop_index)+1200]
    except IndexError:
        return None

    pad = 16000 - len(waveform)
    if pad < 0:
        return None

    left_pad = pad // 2
    right_pad = pad - left_pad
    return np.pad(waveform, ((left_pad, right_pad), (0, 0)), 'constant')


def record_word(file_name):
    fs = 16000
    seconds = 2
    print("record start!")
    waveform = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    print("record stop!")

    waveform = get_word(waveform)
    if waveform is not None:
        write(file_name, fs, waveform)


def get_record(path, label, speaker_id):
    num = str(len(os.listdir(path))).zfill(3)
    file_name = os.path.join(path, str(speaker_id) + '_' + num + '.wav')

    print_double()
    print("Next word: " + label)
    sleep(2)
    record_word(file_name)
    print_double()


def main():
    speaker_id = 0
    if len(sys.argv) > 1:
        speaker_id = int(sys.argv[1])

    print("Are you ready??")
    sleep(3)
    test_set_path = '..\\test_set'
    if not os.path.exists(test_set_path):
        os.mkdir(test_set_path)

    for label in LABELS:
        path = os.path.join(test_set_path, label)
        if not os.path.exists(path):
            os.mkdir(path)

        get_record(path, label, speaker_id)


if __name__ == '__main__':
    main()
    print_double()
    print("Thank you!")

    # sleep(3)
    # import soundfile as sf
    #
    # filename = 'test_set\\tree\\000.wav'
    # # Extract data and sampling rate from file
    # data, fs = sf.read(filename, dtype='float32')
    # sd.play(data, fs)
    # status = sd.wait()  # Wait until file is done playing()
