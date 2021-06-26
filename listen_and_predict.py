import torch

from model_manager import ModelManager
from audio_utils.audio_recorder import AudioRecorder

MODEL_NAME = 'trained_models\\checkpoint.pt'


def get_predictions(words):
    model_manager = ModelManager()
    model_manager.load_model(MODEL_NAME)

    predictions = []
    for word in words:
        waveform = torch.from_numpy(word).unsqueeze(0).float()
        out = model_manager.predict(waveform).squeeze()
        max_i = out.argmax(dim=-1)
        prediction = model_manager.index_to_label(max_i)
        predictions.append(prediction)

    print(f"Did you say {', '.join(predictions)}?")


def main():
    ar = AudioRecorder()
    ar.listen()

    words = ar.get_words()
    get_predictions(words)


if __name__ == '__main__':
    main()
