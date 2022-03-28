import tensorflow as tf

from agent.get_shape import get_shape


class MetaMultiObj:
    def __init__(self, funcs, alpha_func, num_objs, alpha_scope, func_scopes):
        self.funcs = funcs
        self.alpha_func = alpha_func
        self.num_objs = num_objs
        self.alpha_scope = alpha_scope
        self.func_scopes = func_scopes

    def __call__(self, features, *args, **kwargs):
        ys = []
        for func, func_scope in zip(self.funcs, self.func_scopes):
            # for instance, ys = [logits, vf, qf]
            y = func(features, func_scope)
            ys.append(y)
        assert len(ys) == self.num_objs
        # if alpha_func is None, make a default mlp function as alpha_func
        if self.alpha_func is None:
            self.alphas = self._alpha_net(features, self.num_objs, self.alpha_scope)
        else:
            self.alphas = self.alpha_func(features, self.num_objs, self.alpha_scope)
        # new_ys should be a weighted sum of each element of ys
        # new_ys should has shape [logits, vf, qf] if y = [logits, vf, qf],
        # where logits = alphas[0] * ys[0][logits] + ... + alphas[k] * ys[k][logits]
        new_ys = self._reweight(ys, self.alphas)
        return new_ys

    def _alpha_net(self, features, num_objs, alpha_scope):
        if features.__class__ == list:
            features = tf.concat(features, axis=-1)
        elif features.__class__ == dict:
            features = tf.concat(list(features.values()), axis=-1)
        net = features
        size = get_shape(net)[-1]
        with tf.variable_scope(alpha_scope, reuse=tf.AUTO_REUSE):
            net = tf.layers.dense(net, size, tf.nn.relu, name="dense_0")
            net = tf.layers.dense(net, size, tf.nn.relu, name="dense_1")
            alpha_logits = tf.layers.dense(net, num_objs, None, name="logits")
            alphas = tf.nn.softmax(alpha_logits, axis=-1)
        return alphas

    @staticmethod
    def _reweight(ys, alphas):
        new_ys = []
        for i in range(len(ys[0])):
            concat_y = tf.concat([tf.expand_dims(y[i], axis=2) for y in ys], axis=2)
            expand_alphas = alphas
            for _ in range(len(get_shape(concat_y)) - len(get_shape(alphas))):
                expand_alphas = tf.expand_dims(alphas, axis=-1)
            new_y = tf.reduce_sum(expand_alphas * concat_y, axis=2)
            new_ys.append(new_y)
        return new_ys
