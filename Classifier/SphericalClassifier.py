from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import InputSpec
from keras import regularizers

class SphericalClassifier(Layer):

    def __init__(self, nClasses, **kwargs):
        super(SphericalClassifier, self).__init__(**kwargs)
        self.nClasses = nClasses
        self.input_spec = [InputSpec(ndim=2)]

    def build(self, input_shape):

        self.W = self.add_weight(shape=(self.nClasses, input_shape[1]),
                                 name='W',
                                 initializer='uniform',
                                 trainable=True)

        super(SphericalClassifier, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nClasses)

    def cosine_similarity(self):
        w = self.W
        print(w)
        print('x : ', x.shape, x.dtype)

        i = K.repeat(x, self.nClasses)

        i = K.l2_normalize(i, axis=-1)
        w = K.l2_normalize(w, axis=-1)
        i = K.print_tensor(i, message='i : ')

        print('i : ', i.shape, i.dtype)
        print('w : ', w.shape, w.dtype)
        K.print_tensor(i)
        w = K.print_tensor(w, message='w : ')

        y = K.sum(i * w, axis=-1, keepdims=False)

        y = K.print_tensor(y, message='y : ')
        print('y : ', y.shape)
        return y

    def call(self, x, mask=None):
        w = self.W

        i = K.repeat(x, self.nClasses)

        d = i-w
        d = K.square(d)
        d = K.sum(d, axis=-1)
        d = K.l2_normalize(d, axis=-1)

        min = K.min(d, axis=-1, keepdims=True)
        min = K.repeat_elements(min, self.nClasses, -1)
        #min = K.ones([self.nClasses])

        #one = K.ones([self.nClasses])
        y = (min) / (d)

        y = K.l2_normalize(y)

        return y

