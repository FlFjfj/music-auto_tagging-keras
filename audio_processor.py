import librosa
import numpy as np
import wave

SR = 12000
N_FFT = 512
N_FFT2 = int(N_FFT / 2) + 1
HOP_LEN = 256
N_MIX = 32
DURA = 15
FRAMES = int(SR * DURA / HOP_LEN - 1)

OUT_BR = 4

melgram_shape = (0, 1, N_FFT2, FRAMES)

def get_spectro(file):
    src, sr = librosa.load(file, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA*SR)

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA*SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[(n_sample-n_sample_fit)//2:(n_sample+n_sample_fit)//2]

    hann = np.hanning(N_FFT)
    spectral = [[0.00000001] * FRAMES for _ in range(N_FFT2)]

    for i in range(FRAMES):
        data = src[i * HOP_LEN:i * HOP_LEN + N_FFT]
        filtred = hann * data
        spect = np.fft.rfft(filtred)
        res = list(map(lambda x: x.real, spect))
        for j in range(N_FFT2):
            spectral[j][i] += res[j]
    res = np.log(np.abs(np.array(spectral)))
    res = res[np.newaxis, np.newaxis, :]
    return res


def sound_from_spectro(spectro):
    in_data = np.exp(np.swapaxes(spectro, 0, 1))
    out_data = []
    for i in range(FRAMES):
        out_data.append(np.fft.irfft(in_data[i]))

    result = list([0][0:N_MIX])

    for i in range(FRAMES - 1):
        result.extend(out_data[i][HOP_LEN:N_FFT - N_MIX])
        left = out_data[i][N_FFT - N_MIX:]
        right = out_data[i+1][HOP_LEN - N_MIX:HOP_LEN]
        for j in range(N_MIX):
            result.append((left[j] + j * right[j]) / (j+1))

    result = np.array(list(map(lambda x: np.int32(x * (1 << (OUT_BR * 8))), result)), dtype=np.int32)
    raw = result.tobytes()

    file = wave.open("output.wav", 'w')
    file.setframerate(SR)
    file.setsampwidth(OUT_BR)
    file.setnchannels(1)
    file.writeframes(raw)
    file.close()
