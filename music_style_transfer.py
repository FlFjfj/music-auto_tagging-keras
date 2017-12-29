import numpy as np
import audio_processor as ap
import tranfer_model as tm
import keras
from keras import backend as K
import tensorflow as tf
from tensorflow import losses as tfl


def main():
    train, loss, target = tm.build_model()

    sess = K.get_session()

    for i in range(200):
        if i % 10 == 0:
            loss_val, _ = sess.run([loss, train], {K.learning_phase(): 0})
            print("Iteration", i, "loss:", loss_val)
        else:
            sess.run(train, {K.learning_phase(): 0})

    result = sess.run(target)
    ap.sound_from_spectro(np.reshape(result, [ap.N_FFT2, ap.FRAMES]))


if __name__ == "__main__":
    main()