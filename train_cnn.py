import time
import numpy as np
from music_tagger_cnn import MusicTaggerCNN
import music_tagger_cnn as cnnmodel
import keras
import prepare_dataset as pd
import audio_processor as ap



def main():
    train_x, train_y = np.load("train_x.npy"), np.load("train_y.npy")
    train_x = np.transpose(train_x, (0, 2, 3, 1))
    model = MusicTaggerCNN()
    model.summary()
    model.compile(keras.optimizers.Adam(), keras.losses.binary_crossentropy, metrics=["accuracy"])
    # predict the tags like this
    print('Training... with melgrams: ', train_x.shape)
    start = time.time()
    model.fit(train_x, train_y, 10, 10, shuffle=True)
    # print like this...
    print("Training is done. It took %d seconds." % (time.time() - start))
    model.save("stupid_model_tensorflow.h5")


if __name__ == '__main__':
    main()
