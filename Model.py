import tensorflow as tf


class Model:
    def __init__(self, numStates, numActions, batchSize):
        self._numStates = numStates
        self._numActions = numActions
        self._batchSize = batchSize

        # define vars
        self._states = None
        self._actions = None
        self._logits = None
        self._optimizer = None
        self._var_init = None  # huh

        self.InitModel()

    def InitModel(self):
        # self._states = tf.placeholder(shape=[None, self._numStates], dtype=tf.float32)
        # self._q_s_a = tf.placeholder(shape=[None, self.numActions], dtype=tf.float32)

        # model = Model(5, 5 , 10)
        pass

W = tf.Variable(tf.ones(shape=(2, 2)), name="W")
b = tf.Variable(tf.zeros(shape=(2)), name="b")


@tf.function
def forward(x):
    return W * x + b


out_a = forward([1, 0])
print(out_a)
print(W)
