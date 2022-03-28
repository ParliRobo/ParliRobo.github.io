import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--num_task", type=int, default=2, help="num task")
parser.add_argument("--num_move", type=int, default=24, help="num move")
parser.add_argument("--num_camera", type=int, default=11, help="num camera")
parser.add_argument("--nof_policy", type=int, default=2, help="number of policies")

parser.add_argument("--use_hrnn", type=bool, default=0, help="whether to use tmp hierarchy rnn (lstm+rmc)")
parser.add_argument("--use_pixel_control", type=bool, default=1, help="whether to use pixel control")

parser.add_argument("--channels", type=int, default=15, help="num of channels of each state")
parser.add_argument("--dims", type=int, default=85, help="dimension of vector of each state")
parser.add_argument("--metas", type=int, default=51, help="dimension of meta of each state")
parser.add_argument("--image_size", type=int, default=42, help="input image size")

parser.add_argument("--vf_clip", type=float, default=1.0, help="clip of value function")
parser.add_argument("--ppo_clip", type=float, default=0.2, help="clip of ppo loss")
parser.add_argument("--gamma", type=float, default=0.997, help="discount rate")
parser.add_argument("--time_scale", type=int, default=4, help="time scale of hierarchy rnn")
parser.add_argument("--pi_coef", type=float, default=20.0, help="weight of policy fn loss")
parser.add_argument("--vf_coef", type=float, default=1.0, help="weight of v value fn loss")
parser.add_argument("--qf_coef", type=float, default=1.0, help="weight of q value fn loss")
parser.add_argument("--ent_coef", type=float, default=0.5, help="weight of entropy loss")
parser.add_argument("--zero_init", type=bool, default=False, help="whether to zero init initial state")
parser.add_argument("--grad_clip", type=float, default=40.0, help="global grad clip")

parser.add_argument("--seed", type=int, default=12358, help="random seed")

kwargs = vars(parser.parse_args())


def warp_Model():
    import tensorflow as tf
    from agent.categorical import categorical
    from agent.entropy import entropy_from_logits
    from agent.drtrace import log_probs_from_logits_and_actions
    from agent.get_shape import get_shape
    from agent.drtrace import from_logits as drtrace_from_logits
    from agent.mmo import MetaMultiObj

    class Model(object):
        def __init__(self,
                     num_task,
                     num_move,
                     num_camera,
                     gamma,
                     rnn,
                     use_hrnn,
                     use_pixel_control,
                     is_training=False,
                     scope="agent",
                     **kwargs):
            self.num_move = num_move
            self.num_camera = num_camera
            self.use_hrnn = use_hrnn

            self.s_vec = kwargs.get("vec")
            self.s_image = kwargs.get("image")
            self.s_meta = kwargs.get("meta")
            self.s_voice = kwargs.get("voice")
            self.a = kwargs.get("actions")
            self.taus = kwargs.get("taus")
            self.state_in = kwargs.get("state_in")
            self.alphas = kwargs.get("alphas")
            self.embed_t = kwargs.get("interval_time")

            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                with tf.variable_scope("ws", reuse=tf.AUTO_REUSE):
                    vision_feature, self.cnn_feature = self.vision_encoder(self.s_image, self.s_voice)
                    meta_feature = self.meta_encoder(self.s_meta)
                    scalar_feature, scalar_context = self.scalar_encoder(
                        tf.concat([self.s_vec, meta_feature], axis=-1))

                    actions = [tf.squeeze(t, axis=-1) for t in tf.split(self.a, 4, axis=-1)]
                    actions_onehot = [
                        tf.one_hot(head_action, depth=head_space, dtype=tf.float32)
                        for head_action, head_space in zip(actions, [2, self.num_move, self.num_camera, 2])]

                    concat_actions_onehot = tf.concat(actions_onehot, axis=-1)

                    self.feature, self.state_out = self.core(
                        vision_feature, scalar_feature, self.embed_t, self.alphas,
                        rnn, concat_actions_onehot, self.state_in)

                    if self.use_hrnn:
                        self.p_zs = self.feature["p_zs"]
                        self.p_mus = self.feature["p_mus"]
                        self.p_sigmas = self.feature["p_sigmas"]
                        self.q_mus = self.feature["q_mus"]
                        self.q_sigmas = self.feature["q_sigmas"]
                        self.feature = self.feature["q_zs"]

                    features = [self.feature, scalar_context, meta_feature, self.embed_t, self.alphas]
                    move_taus, fight_taus = tf.split(self.taus, 2, axis=-1)
                    move_logits, move_qf, self.move_vf = self.a_net(features, move_taus, "move_a_net")
                    fight_logits, fight_qf, self.fight_vf = self.a_net(features, fight_taus, "fight_a_net")

                # self.alphas, alpha_logits = self.alpha_net(features, 2, "ALPHAS")
                # self.alpha_entropy = tf.reduce_mean(entropy_from_logits(alpha_logits))

                tf.summary.scalar("move_alpha", tf.reduce_mean(self.alphas[:, :, 0]))
                tf.summary.scalar("fight_alpha", tf.reduce_mean(self.alphas[:, :, 1]))

                with tf.variable_scope("ws", reuse=tf.AUTO_REUSE):
                    # a list of 4 heads
                    logits = MetaMultiObj._reweight(
                        [move_logits, fight_logits], self.alphas)
                    self.current_act_logits = tf.concat(logits, axis=-1)
                    self.current_act = tf.concat(
                        [categorical(head_logits) for head_logits in logits], axis=-1)

                    if is_training:
                        self.r = kwargs.get("rewards")
                        self.r_move = kwargs.get("move") * self.alphas[:, :, 0]
                        self.r_fight = kwargs.get("fight") * self.alphas[:, :, 1]

                        old_logits = kwargs.get("logits")[:, :-1, :]
                        old_logits = tf.split(
                            old_logits, [2, self.num_move, self.num_camera, 2], axis=-1)

                        self.mask = tf.cast(kwargs.get("mask"), tf.float32)
                        move_mask = self.mask[:, :-1] * tf.cast(
                            actions[0][:, 1:] > 0, tf.float32)
                        # list
                        self.masks = [self.mask[:, :-1], move_mask,
                                      self.mask[:, :-1], self.mask[:, :-1]]

                        self.current_value = self.v_net(self.feature, meta_feature) * self.mask

                        # list
                        log_rhos = [
                            log_probs_from_logits_and_actions(
                                head_logits[:, :-1, :], head_action[:, 1:]) - tf.maximum(
                                log_probs_from_logits_and_actions(
                                    head_old_logits, head_action[:, 1:]),
                                tf.math.log(1e-8)) * head_mask
                            for head_logits, head_action, head_old_logits, head_mask
                            in zip(logits, actions, old_logits, self.masks)]

                        # list
                        self.rhos = [tf.exp(head_log_rhos) for head_log_rhos in log_rhos]

                        vtraces = [
                            drtrace_from_logits(
                                behaviour_policy_logits=head_old_logits,
                                target_policy_logits=head_logits[:, :-1],
                                actions=head_action[:, 1:],
                                discounts=gamma * tf.ones_like(self.r[:, 1:], tf.float32),
                                rewards=self.r[:, 1:],
                                v_values=self.current_value[:, :-1],
                                q_values=self.current_value[:, :-1],
                                target_values=self.current_value[:, :-1],
                                bootstrap_value=self.current_value[:, -1],
                                laser_threshold=0.1)
                            for head_old_logits, head_logits, head_action
                            in zip(old_logits, logits, actions)
                        ]
                        self.advantages = [vtrace.advantages for vtrace in vtraces]
                        self.vs = tf.reduce_mean([vtrace.vs for vtrace in vtraces], axis=0)

                        # # --------------------------------------------------
                        #
                        # move_qa = [tf.reduce_sum(head_qf[:, :-1] * head_action_onehot[:, 1:], axis=-1) * head_mask
                        #            for head_qf, head_action_onehot, head_mask in
                        #            zip(move_qf, actions_onehot, self.masks)]
                        # fight_qa = [tf.reduce_sum(head_qf[:, :-1] * head_action_onehot[:, 1:], axis=-1) * head_mask
                        #             for head_qf, head_action_onehot, head_mask in
                        #             zip(fight_qf, actions_onehot, self.masks)]
                        # move_vf = self.move_vf * self.mask
                        # fight_vf = self.fight_vf * self.mask
                        #
                        # self.move_mix_qa = MixingNet(
                        #     agent_num=4, hidden_dim=32, qmix_hidden_dim=32)(
                        #     feature=self.feature[:, :-1, :], qs=move_qa)
                        # self.fight_mix_qa = MixingNet(
                        #     agent_num=4, hidden_dim=32, qmix_hidden_dim=32)(
                        #     feature=self.feature[:, :-1, :], qs=fight_qa)
                        #
                        # move_gae = gae(
                        #     lambdas=0.95 * tf.ones_like(self.r_move[:, 1:], tf.float32),
                        #     discounts=gamma * tf.ones_like(self.r_move[:, 1:], tf.float32),
                        #     rewards=self.r_move[:, 1:],
                        #     v_values=move_vf[:, :-1],
                        #     q_values=self.move_mix_qa,
                        #     target_values=move_vf[:, :-1],
                        #     bootstrap_value=move_vf[:, -1],
                        # )
                        # self.move_mix_qs = move_gae.qs
                        # self.move_vs = move_gae.vs
                        #
                        # fight_gae = gae(
                        #     lambdas=0.95 * tf.ones_like(self.r_fight[:, 1:], tf.float32),
                        #     discounts=gamma * tf.ones_like(self.r_fight[:, 1:], tf.float32),
                        #     rewards=self.r_fight[:, 1:],
                        #     v_values=fight_vf[:, :-1],
                        #     q_values=self.fight_mix_qa,
                        #     target_values=fight_vf[:, :-1],
                        #     bootstrap_value=fight_vf[:, -1],
                        # )
                        # self.fight_mix_qs = fight_gae.qs
                        # self.fight_vs = fight_gae.vs

                        adv = tf.reduce_mean([self.advantages], axis=0)
                        adv_mean = tf.reduce_mean(adv)
                        adv = adv - adv_mean
                        adv_std = tf.math.sqrt(tf.reduce_mean(adv ** 2))

                        tf.summary.scalar("adv_mean", adv_mean)
                        tf.summary.scalar("adv_std", adv_std)

                        self.entropy = tf.reduce_sum(
                            [entropy_from_logits(head_logits) for head_logits in logits],
                            axis=0) * self.mask

                        if use_pixel_control:
                            self.feature_prediction, self.reward_prediction = self.control_net(
                                self.feature[:, :-1, :], concat_actions_onehot[:, 1:, :], meta_feature[:, :-1])

        def get_current_act(self):
            return self.current_act

        def get_current_act_logits(self):
            return self.current_act_logits

        def a_net(self, features, taus, scope="a_net"):
            feature, context, meta, embed_t, alphas = features
            meta = tf.concat([meta, embed_t, alphas], axis=-1)
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                tau_whether_move, tau_move, tau_camera, tau_shoot = tf.split(taus, 4, axis=-1)
                vf = self.v_net(feature, meta)
                context = tf.layers.dense(
                    tf.concat([context, meta], axis=-1),
                    64, tf.nn.relu, name="context")
                feature = tf.concat([feature, meta], axis=-1)
                feature = tf.layers.dense(
                    feature, 64, activation=tf.nn.relu, name="dense")
                prelogits = self.layer_norm(feature + tf.layers.dense(
                    feature, 64, activation=tf.nn.relu, name="prelogits"), "prelogits_ln")
                logits = self.glu(prelogits, context, 2 + self.num_move + self.num_camera + 2)
                whether_move_logits, move_logits, camera_logits, shoot_logits = tf.split(
                    logits, [2, self.num_move, self.num_camera, 2], axis=-1)
                whether_move_qf, whether_move_logits = self._casa_(vf, whether_move_logits, tau_whether_move)
                move_qf, move_logits = self._casa_(vf, move_logits, tau_move)
                camera_qf, camera_logits = self._casa_(vf, camera_logits, tau_camera)
                shoot_qf, shoot_logits = self._casa_(vf, shoot_logits, tau_shoot)

            return ([whether_move_logits, move_logits, camera_logits, shoot_logits],
                    [whether_move_qf, move_qf, camera_qf, shoot_qf], vf)

        def move_head(self, feature, context, meta, vf, tau, scope="move"):
            net = tf.concat([feature, meta], axis=-1)
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                context = tf.layers.dense(
                    tf.concat([context, meta], axis=-1),
                    32, tf.nn.relu, name="context")
                net = tf.layers.dense(
                    net,
                    16,
                    activation=tf.nn.relu,
                    name="dense")
                theta_prelogits = self.layer_norm(
                    net + tf.layers.dense(
                        net,
                        16,
                        activation=tf.nn.relu,
                        name="theta_prelogits"),
                    "theta_prelogits_ln")
                theta_logits = self.glu(theta_prelogits, context,
                                        self.num_move // 2, "theta_logits")
                theta_logits = theta_logits - tf.reduce_mean(
                    theta_logits, axis=-1, keepdims=True)
                # theta_logits = self.tile(theta_logits)
                r_prelogits = self.layer_norm(
                    net + tf.layers.dense(
                        net,
                        16,
                        activation=tf.nn.relu,
                        name="r_prelogits"),
                    "r_prelogits_ln")
                r_logits = self.glu(r_prelogits, theta_prelogits,
                                    2, "r_logits")
                r_logits = r_logits - tf.reduce_mean(
                    r_logits, axis=-1, keepdims=True)
                move_logits = theta_logits[:, :, None, :] + r_logits[:, :, :, None]
                move_logits = tf.reshape(
                    move_logits, get_shape(move_logits)[:-2] + [self.num_move])
                qf, move_logits = self._casa_(vf, move_logits, tau)
            return move_logits, qf

        def whether_move_head(self, feature, context, meta, vf, tau, scope="whether_move"):
            net = tf.concat([feature, meta], axis=-1)
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                context = tf.layers.dense(
                    tf.concat([context, meta], axis=-1),
                    16, tf.nn.relu, name="ar_embedding")
                net = tf.layers.dense(
                    net,
                    16,
                    activation=tf.nn.relu,
                    name="dense")
                whether_move_prelogits = self.layer_norm(
                    net + tf.layers.dense(
                        net,
                        16,
                        activation=tf.nn.relu,
                        name="whether_move_prelogits"),
                    "whether_move_prelogits_ln")
                whether_move_logits = self.glu(whether_move_prelogits, context,
                                               2, "whether_move_logits")
                whether_move_logits = whether_move_logits - tf.reduce_mean(
                    whether_move_logits, axis=-1, keepdims=True)
                qf, whether_move_logits = self._casa_(vf, whether_move_logits, tau)
            return whether_move_logits, qf

        def camera_head(self, feature, context, meta, vf, tau, scope="camera"):
            net = tf.concat([feature, meta], axis=-1)
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                context = tf.layers.dense(
                    tf.concat([context, meta], axis=-1),
                    16, tf.nn.relu, name="ar_embedding")
                net = tf.layers.dense(
                    net,
                    16,
                    activation=tf.nn.relu,
                    name="dense")
                camera_prelogits = self.layer_norm(
                    net + tf.layers.dense(
                        net,
                        16,
                        activation=tf.nn.relu,
                        name="camera_prelogits"),
                    "camera_prelogits_ln")
                camera_logits = self.glu(camera_prelogits, context,
                                         self.num_camera, "camera_logits")
                camera_logits = camera_logits - tf.reduce_mean(
                    camera_logits, axis=-1, keepdims=True)
                qf, camera_logits = self._casa_(vf, camera_logits, tau)
            return camera_logits, qf

        def shoot_head(self, feature, context, meta, vf, tau, scope="shoot"):
            net = tf.concat([feature, meta], axis=-1)
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                context = tf.layers.dense(
                    tf.concat([context, meta], axis=-1),
                    16, tf.nn.relu, name="ar_embedding")
                net = tf.layers.dense(
                    net,
                    16,
                    activation=tf.nn.relu,
                    name="dense")
                shoot_prelogits = self.layer_norm(
                    net + tf.layers.dense(
                        net,
                        16,
                        activation=tf.nn.relu,
                        name="shoot_prelogits"),
                    "shoot_prelogits_ln")
                shoot_logits = self.glu(shoot_prelogits, context,
                                        2, "shoot_logits")
                shoot_logits = shoot_logits - tf.reduce_mean(
                    shoot_logits, axis=-1, keepdims=True)
                qf, shoot_logits = self._casa_(vf, shoot_logits, tau)
            return shoot_logits, qf

        def alpha_net(self, features, num_objs, scope="alpha_net"):
            feature, context, meta = features
            net = tf.concat([feature, meta], axis=-1)
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                context = tf.layers.dense(
                    tf.concat([context, meta], axis=-1),
                    16, tf.nn.relu, name="ar_embedding")
                logits = self.glu(net, context,
                                  num_objs, "alpha_logits")
                logits = logits - tf.reduce_mean(
                    logits, axis=-1, keepdims=True)
                alphas = tf.nn.softmax(logits)
            return alphas, logits

        def v_net(self, feature, meta, scope="v_value"):
            net = tf.concat([feature, meta], axis=-1)
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                meta = tf.layers.dense(
                    meta, get_shape(meta)[-1],
                    activation=tf.nn.relu, name="meta_dense")
                net = tf.layers.dense(
                    net, 16, tf.nn.relu, name="dense")
                net = self.layer_norm(
                    net + tf.layers.dense(
                        net,
                        get_shape(net)[-1],
                        activation=tf.nn.relu,
                        name="dense_1"))
                net = self.glu(net, meta, 16)
                v_value = tf.squeeze(
                    tf.layers.dense(
                        net,
                        1,
                        activation=None,
                        name="v_value"),
                    axis=-1)
            return v_value

        # fixme add mixing_net
        def mixing_net(self, feature, qas, scope='mixing_net'):
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                hyper_w1_dense = tf.layers.dense(feature, 32, tf.nn.relu, name='w1_dense1')
                w1 = tf.layers.dense(hyper_w1_dense, 4 * 32, None, name='w1_dense2')
                hyper_w2_dense = tf.layers.dense(feature, 32, tf.nn.relu, name='w2_dense1')
                w2 = tf.layers.dense(hyper_w2_dense, 32, None, name='w2_dense2')
                b1 = tf.layers.dense(feature, 32, None, name='b1_dense1')
                hyper_b2_dense = tf.layers.dense(feature, 32, tf.nn.relu, name='b2_dense1')
                b2 = tf.layers.dense(hyper_b2_dense, 1, None, name='b2_dense2')

                B, T, feature_dim = get_shape(feature)
                qas = tf.transpose(tf.convert_to_tensor(qas, tf.float32), perm=[1, 2, 0])
                # qas = tf.reshape(qas, [B, T, 1, 4])
                qas = tf.expand_dims(qas, 2)
                w1 = tf.reshape(tf.math.abs(w1, name='abs_w1'), (B, T, 4, 32))
                w2 = tf.reshape(tf.math.abs(w2, name='abs_w2'), (B, T, 1, 32))

                b1 = tf.reshape(b1, (B, T, 1, 32))
                b2 = tf.reshape(b2, (B, T, 1, 1))

                q_total = tf.squeeze(tf.matmul((tf.matmul(qas, w1) + b1), w2, transpose_b=True) + b2, axis=[-2, -1])
            return q_total

        def vision_encoder(self, image, voice, scope="vision_encoder"):
            image = 0.01 * tf.cast(image, tf.float32)
            image, mini_map = tf.split(image, [-1, 4], axis=-1)

            voice = 0.01 * tf.cast(voice, tf.float32)

            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                with tf.variable_scope("vision"):
                    feature_0, cnn_feature = self._cnn(image, [16, 32, 32, 32])
                with tf.variable_scope("mini_map"):
                    feature_1, _ = self._cnn(mini_map, [4, 8, 8, 8])
                with tf.variable_scope("voice"):
                    feature_2, _ = self._voice_cnn(voice, [2, 2, 2, 2])
                feature = tf.layers.dense(
                    tf.concat([feature_0, feature_1, feature_2], axis=-1),
                    192, tf.nn.relu, name="vision_feature")
            return feature, cnn_feature

        @staticmethod
        def _cnn(image, filter):
            shape = get_shape(image)

            image = tf.reshape(image, [-1] + shape[-3:])
            kernel = [7, 5, 3, 3]
            stride = [3, 2, 1, 1]

            image = tf.layers.conv2d(
                image,
                filters=filter[0],
                kernel_size=kernel[0],
                strides=stride[0],
                padding="valid",
                activation=tf.nn.relu,
                name="conv_%d" % 0)
            image = tf.layers.conv2d(
                image,
                filters=filter[1],
                kernel_size=kernel[1],
                strides=stride[1],
                padding="valid",
                activation=tf.nn.relu,
                name="conv_%d" % 1)
            image += tf.layers.conv2d(
                image,
                filters=filter[2],
                kernel_size=kernel[2],
                strides=stride[2],
                padding="same",
                activation=None,
                name="conv_%d" % 2)
            image += tf.layers.conv2d(
                tf.nn.relu(image),
                filters=filter[3],
                kernel_size=kernel[3],
                strides=stride[3],
                padding="same",
                activation=None,
                name="conv_%d" % 3)
            image = tf.nn.relu(image)

            new_shape = get_shape(image)
            cnn_feature = tf.reshape(
                image,
                [shape[0], shape[1], new_shape[1], new_shape[2], new_shape[3]])

            feature = tf.reshape(
                image,
                [shape[0], shape[1], new_shape[1]
                 * new_shape[2] * new_shape[3]])
            return feature, cnn_feature

        @staticmethod
        def _voice_cnn(image, filter):
            shape = get_shape(image)

            image = tf.reshape(image, [-1] + shape[-3:])
            kernel = [5, 4, 3, 3]
            stride = [1, 1, 1, 1]
            print(image)
            image = tf.layers.conv2d(
                image,
                filters=filter[0],
                kernel_size=[kernel[0], kernel[0] * 4],
                strides=[stride[0], stride[0] * 4],
                padding="valid",
                activation=tf.nn.relu,
                name="conv_%d" % 0)
            print(image)
            image = tf.layers.conv2d(
                image,
                filters=filter[1],
                kernel_size=[kernel[1], kernel[1] * 2],
                strides=[stride[1], stride[1] * 2],
                padding="valid",
                activation=tf.nn.relu,
                name="conv_%d" % 1)
            print(image)
            image += tf.layers.conv2d(
                image,
                filters=filter[2],
                kernel_size=[kernel[2], kernel[2] * 2],
                strides=[stride[2], stride[2]],
                padding="same",
                activation=None,
                name="conv_%d" % 2)
            print(image)
            image += tf.layers.conv2d(
                tf.nn.relu(image),
                filters=filter[3],
                kernel_size=[kernel[3], kernel[3] * 2],
                strides=[stride[3], stride[3]],
                padding="same",
                activation=None,
                name="conv_%d" % 3)
            image = tf.nn.relu(image)
            print(image)
            new_shape = get_shape(image)
            cnn_feature = tf.reshape(
                image,
                [shape[0], shape[1], new_shape[1], new_shape[2], new_shape[3]])

            feature = tf.reshape(
                image,
                [shape[0], shape[1], new_shape[1]
                 * new_shape[2] * new_shape[3]])
            return feature, cnn_feature

        def scalar_encoder(self, vec, scope="scalar_encoder"):
            size = get_shape(vec)[-1]
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                scalar = self.layer_norm(vec, "ln_0")
                scalar = tf.layers.dense(
                    scalar, size, tf.nn.relu, name="dense_0")
                scalar = tf.layers.dense(
                    scalar, size, None, name="dense_1")
                context = self.layer_norm(vec + scalar, "ln_0")
                scalar = tf.layers.dense(
                    scalar, 64, tf.nn.relu, name="dense_2")
            return scalar, context

        def meta_encoder(self, meta, scope="meta_encoder"):
            size = get_shape(meta)[-1]
            context = meta
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                context = tf.layers.dense(
                    context, size, tf.nn.relu, name="dense_0")
                context = tf.layers.dense(
                    context, size, None, name="dense_1")
                context = self.layer_norm(meta + context, "ln")
            return context

        def core(self, vision, scalar, embed_t, alphas, rnn, prev_a, state_in, scope="core"):
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                feature = tf.concat(
                    [vision, scalar, embed_t, alphas, prev_a], axis=-1)
                feature = tf.layers.dense(
                    feature, 192, tf.nn.relu, name="feature")

                if self.use_hrnn:
                    initial_state = tf.split(state_in, [1, -1], axis=-1)
                    feature, count_out, state_out = rnn(
                        feature, initial_state=initial_state)
                    state_out = tf.concat([count_out, state_out], axis=-1)
                else:
                    initial_state = tf.split(state_in, 2, axis=-1)
                    feature, c_out, h_out = rnn(
                        feature, initial_state=initial_state)
                    state_out = tf.concat([c_out, h_out], axis=-1)
            return feature, state_out

        def control_net(self, feature, a_onehot, meta, scope="control"):
            shape = get_shape(feature)
            batch_size, seq_length = shape[0], shape[1]
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                a_embedding = tf.layers.dense(
                    a_onehot, 32, use_bias=False, name="a_embedding")
                feature_with_a = tf.concat([feature, a_embedding], axis=-1)

                with tf.variable_scope("dynamic", reuse=tf.AUTO_REUSE):
                    feature_after_dynamic = tf.layers.dense(
                        feature_with_a, 128, activation=tf.nn.relu)
                    feature_after_dynamic = feature + tf.layers.dense(
                        feature_after_dynamic, 128,
                        activation=tf.nn.relu, name="dynamic")

                with tf.variable_scope("feature_prediction", reuse=tf.AUTO_REUSE):
                    feature_prediction = tf.layers.dense(
                        feature_after_dynamic, 128,
                        activation=None, name="feature_prediction")
                    feature_prediction = tf.nn.l2_normalize(feature_prediction, axis=-1)

                with tf.variable_scope("reward_prediction", reuse=tf.AUTO_REUSE):
                    reward_prediction = tf.squeeze(self.glu(feature_after_dynamic, meta, 1))

            return feature_prediction, reward_prediction

        @staticmethod
        def _casa_(vf, h, tau):
            p = tf.stop_gradient(tf.nn.softmax(tau * h, axis=-1))
            E_h = tf.reduce_sum(h * p, axis=-1, keepdims=True)
            qf = tf.stop_gradient(tf.expand_dims(vf, axis=-1)) + h - E_h
            return qf, tau * h

        @staticmethod
        def tile(tensor):
            shift_left = tf.concat(tf.split(tensor, [1, -1], axis=-1)[::-1], axis=-1)
            shift_right = tf.concat(tf.split(tensor, [-1, 1], axis=-1)[::-1], axis=-1)
            return (shift_left * 0.25 + tensor + shift_right * 0.25) / 1.5

        @staticmethod
        def glu(inputs, context, output_size, scope="glu"):
            input_size = get_shape(inputs)[-1]
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                gate = tf.layers.dense(context, input_size, activation=tf.nn.sigmoid)
                gated_input = gate * inputs
                output = tf.layers.dense(gated_input, output_size)
            return output

        @staticmethod
        def layer_norm(input_tensor, name=None):
            """Run layer normalization on the last dimension of the tensor."""
            return tf.contrib.layers.layer_norm(
                inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)

        def inf(self, sess, input_d):
            fd = dict()
            fd[self.s_image] = input_d["image"]
            fd[self.s_voice] = input_d["voice"]
            fd[self.s_vec] = input_d["vec"]
            fd[self.s_meta] = input_d["meta"]
            fd[self.a] = input_d["actions"]
            fd[self.state_in] = input_d["state_in"]
            fd[self.alphas] = input_d["alphas"]
            fd[self.taus] = input_d["taus"]
            fd[self.embed_t] = input_d["interval_time"]

            a, a_logits, state_in = sess.run(
                [self.current_act, self.current_act_logits,
                 self.state_out],
                feed_dict=fd)

            return {"a": a, "a_logits": a_logits,
                    "state_in": state_in}

    return Model


def build_evaluator_model(kwargs):
    Model = warp_Model()
    import tensorflow as tf
    from agent.TmpHierRNN import TmpHierRNN
    from agent.NewLSTM import NewLSTM
    channels = kwargs["channels"]
    dims = kwargs["dims"]
    metas = kwargs["metas"]
    image_size = kwargs["image_size"]
    num_task = kwargs["num_task"]
    num_move = kwargs["num_move"]
    num_camera = kwargs["num_camera"]
    gamma = kwargs["gamma"]
    time_scale = kwargs["time_scale"]
    state_size = kwargs["state_size"]
    use_hrnn = kwargs["use_hrnn"]
    use_pixel_control = kwargs["use_pixel_control"]
    scope = kwargs.get("scope", "agent")

    phs = dict()

    phs["image"] = tf.placeholder(
        dtype=tf.uint8, shape=[None, None, image_size, image_size, channels])
    phs["voice"] = tf.placeholder(
        dtype=tf.uint8, shape=[None, None, 10, 2 * image_size, 1])
    phs["vec"] = tf.placeholder(dtype=tf.float32, shape=[None, None, dims])
    phs["meta"] = tf.placeholder(dtype=tf.float32, shape=[None, None, metas])
    phs["actions"] = tf.placeholder(dtype=tf.int32, shape=[None, None, 4])
    phs["taus"] = tf.placeholder(dtype=tf.float32, shape=[None, None, 4 * kwargs["num_task"]])
    phs["rewards"] = tf.placeholder(dtype=tf.float32, shape=[None, None])
    phs["state_in"] = tf.placeholder(dtype=tf.float32, shape=[None, state_size])
    phs["alphas"] = tf.placeholder(dtype=tf.float32, shape=[None, None, 2])
    phs["interval_time"] = tf.placeholder(dtype=tf.float32, shape=[None, None, 8])

    if use_hrnn:
        rnn = TmpHierRNN(time_scale, 32, 4, 2, 8, 'lstm', 'rmc',
                         return_sequences=True, return_state=True, name="hrnn")
    else:
        rnn = NewLSTM(
            128, return_sequences=True, return_state=True, name="lstm")

    model = Model(num_task, num_move, num_camera,
                  gamma, rnn, use_hrnn, use_pixel_control, False, scope, **phs)

    return model


if __name__ == '__main__':
    kwargs["state_size"] = 256
    model = build_evaluator_model(kwargs)
