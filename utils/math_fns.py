import tensorflow as tf

def huber(dist, delta):
    return tf.reduce_sum(tf.square(delta) * ( tf.pow(1 + tf.square(dist  / delta), 0.5) - 1 ))

def euclid(dist, axis=0):
    return tf.norm(dist, axis=axis)

def clip_by_global_norm(t_list, clip_norm):
    '''Clips the tensors in the list of tensors *t_list* globally by their norm. This preserves the 
    relative weights of gradients if used on gradients. The inbuilt clip_norm argument of 
    keras optimizers does NOT do this. Global norm clipping is the correct way of implementing
    gradient clipping. The function *tf.clip_by_global_norm()* changes the structure of the passed tensor
    sometimes. This is why I decided not to use it.
    :param t_list: List of tensors to be clipped.
    :param clip_norm: Norm over which the tensors should be clipped.
    :return t_list: List of clipped tensors. 
    :return norm: New norm after clipping.'''
    norm = get_norm(t_list)
    if norm > clip_norm:
        t_list = [tf.scalar_mul(clip_norm / norm, t) for t in t_list]
        norm = clip_norm
    return t_list, norm

def get_norm(t_list):
    return tf.math.sqrt(sum([tf.reduce_sum(tf.square(t)) for t in t_list]))
