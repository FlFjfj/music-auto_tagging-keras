import librosa
import numpy as np
import audio_processor as ap
import music_tagger_cnn as cnnmodel
import os

dataset_dir = "./genres/"
max_files = 10
def main():
    train_x = np.zeros(ap.melgram_shape )
    train_y = np.zeros((0, len(cnnmodel.tags)), dtype=np.bool)
    for tag_idx in range(len(cnnmodel.tags)):
        cur_dir = dataset_dir + cnnmodel.tags[tag_idx] + "/"
        files = os.listdir(dataset_dir + cnnmodel.tags[tag_idx])
        fileidx = 0
        for file in files:
            if fileidx == max_files:
                break
            print("Preparing %s" % cur_dir + file)
            melgram = ap.get_spectro(cur_dir + file)
            train_x = np.concatenate((train_x, melgram), axis=0)
            answer = np.zeros((1,len(cnnmodel.tags)), dtype=np.bool)
            answer[0][tag_idx] = True
            train_y = np.concatenate((train_y, answer), axis=0)
            fileidx += 1

    np.save("train_x.npy", train_x)
    np.save("train_y.npy", train_y)


if __name__ == '__main__':
    main()