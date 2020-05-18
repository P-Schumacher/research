import tensorflow as tf

def huber(dist, delta):
    return tf.reduce_sum(tf.square(delta) * ( tf.pow(1 + tf.square(dist  / delta), 0.5) - 1 ))

def euclid(dist, axis=0):
    return tf.norm(dist, axis=axis)


def huber_not_reduce(dist, delta):
    return tf.square(delta) * ( tf.pow(1 + tf.square(dist  / delta), 0.5) - 1 )
