import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

import audio_processor as ap


def main():
    mel = ap.get_spectro("style.au")
    ap.sound_from_spectro(mel[0][0])

    plt.figure(figsize=(14, 8))
    librosa.display.specshow(mel[0][0], sr=ap.SR, x_axis='time', y_axis='mel')
    plt.title('mel power spectrogram')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()