import tensorflow as tf

def huber(dist, delta):
    return tf.reduce_sum(tf.square(delta) * ( tf.pow(1 + tf.square(dist  / delta), 0.5) - 1 ))

def euclid(dist):
    return tf.norm(dist)
