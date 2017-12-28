import time
import numpy as np
from keras import backend as K
from music_tagger_cnn import MusicTaggerCNN
import keras
import prepare_dataset as pd
import audio_processor as ap



def main():
    train_x, train_y = np.load("train_x.npy"), np.load("train_y.npy")
    model = MusicTaggerCNN()
    model.summary()
    model.compile(keras.optimizers.Adam(), keras.losses.binary_crossentropy)
    # predict the tags like this
    print('Training... with melgrams: ', train_x.shape)
    start = time.time()
    model.fit(train_x, train_y, 10, 10)
    # print like this...
    print("Prediction is done. It took %d seconds." % (time.time() - start))


if __name__ == '__main__':
    main()
