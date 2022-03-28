# coding: utf-8

import copy
import math
import numpy as np
import time

from _math import _power_embed, _soft_rot_embed, _sqrt_embed, _binary_embed, _clip


def enemy_location(players_location, players_rotation, dis_filter, angle_filter):
    """
    :param players_location: (n, 2,)
    :param players_rotation: (n,)
    :param dis_filter: float
    :param angle_filter: float
    :return: dis: (n, n,), angle: (n, n,), shoot_index: (n,)
    """
    # get agent num
    num_agent = len(players_location)
    # calculate Euclidean distance of each pair of agents
    loc_from = np.tile(players_location, (num_agent, 1))
    loc_to = np.reshape(np.tile(players_location, num_agent), (num_agent * num_agent, 2))
    dis = np.sqrt(np.sum(np.square(loc_from - loc_to), axis=-1))
    dis = np.reshape(dis, [num_agent, num_agent])
    # filter by distance
    dis_mask = np.where(dis > dis_filter, 1, 0) + np.eye(num_agent)
    # calculate angle of each pair of agents and rotate coordinate system
    rel_pos_x, rel_pos_y = np.transpose(loc_from - loc_to)
    rel_angle = -np.arctan2(rel_pos_y, rel_pos_x) + np.pi / 2
    # rel_angle[rel_angle < 0] = np.abs(rel_angle[rel_angle < 0]) + 270
    rel_angle %= (2 * np.pi)
    rota_angle = np.repeat(players_rotation, num_agent)
    # calculate the angle of view
    angle = np.abs(rel_angle - rota_angle)
    angle[angle > np.pi] = 2 * np.pi - angle[angle > np.pi]
    angle = np.reshape(angle, [num_agent, num_agent])
    angle = angle * (np.ones_like(angle) - np.eye(num_agent))
    # get shoot_index
    shoot_index = []
    filter_angle = angle + dis_mask * 3 * np.pi
    for agent_i, fa in zip(range(num_agent), filter_angle):
        if min(fa) < angle_filter:
            shoot_index.append(np.argmin(fa))
        else:
            shoot_index.append(-1)

    return dis, angle, shoot_index


def hit_rate_func(dist, angle):
    assert angle >= 0
    dist = max(0.0, min(50.0, dist))
    best_dist = 7.5
    if dist < best_dist:
        dist_rate = 1.0 - ((best_dist - dist) / best_dist) ** 0.8 * 0.7
    else:
        dist_rate = 1.0 - ((dist - best_dist) / (50.0 - best_dist)) ** 0.5 * 0.8
    angle = max(0.0, min(np.pi / 6, angle))
    angle_rate = 1.0 - 0.5 * (angle / (np.pi / 6)) ** 0.5
    return dist_rate * angle_rate


def _rescale(x):
    if x <= 0:
        return x * 2
    return x / 2.5


def preprocessor(shape=(42, 42), landscape_n_values=6):
    def decorator(cls):
        class XGameEnv(cls):
            def __init__(self, *args, **kwargs):
                self.trace_memory_range = {0: [0.1, 0.5]}
                self.trace_length_range = {0: [5.0, 20.0]}
                self.hit_rate_range = {0: [0.8, 0.9]}

                super(XGameEnv, self).__init__(*args, **kwargs)
                self.rBuffRange = {0: [0.1, 1.0],
                                   1: [0.2, 2.0]}
                self.rBoxRange = {0: [0.1, 1.0],
                                  1: [0.2, 2.0]}
                # self.rBulletRange = {0: [0.02, 0.2],
                #                      1: [0.02, 0.2]}
                # self.rWeaponRange = {0: [1.0, 2.0],
                #                      1: [1.0, 2.0]}
                self.rGetCrystalRange = {0: [0.75, 2.25],
                                         1: [0.75, 2.25]}
                self.rLoseCrystalRange = {0: [0.01, 0.1],
                                          1: [0.01, 0.1]}
                self.rKillRange = {0: [30.0, 60.0],
                                   1: [20.0, 40.0]}
                self.rDeathRange = {0: [5, 10],
                                    1: [12.5, 25]}
                # self.rHpRange = {0: [0.0002, 0.02],
                #                  1: [0.0002, 0.02]}
                # self.rShieldRange = {0: [0.0001, 0.01],
                #                      1: [0.0001, 0.01]}
                self.rHurtRange = {0: [0.5, 5.0],
                                   1: [0.25, 2.5]}
                self.rBeHurtRange = {0: [0.05, 0.25],
                                     1: [0.1, 1.0]}
                self.rInterruptMoveRange = {0: [0.01, 0.1],
                                            1: [0.01, 0.1]}
                self.rFaceMoveRange = {0: [0.005, 0.05],
                                       1: [0.005, 0.05]}
                self.rContinuousMoveRange = {0: [0.001, 0.01],
                                             1: [0.001, 0.01]}
                # self.rFinishMoveRange = {0: [0.0001, 0.001],
                #                          1: [0.0001, 0.001]}
                self.rShootInShootRange = {0: [0.01, 0.1],
                                           1: [0.01, 0.1]}
                self.rShootInSightRange = {0: [0.0001, 0.001],
                                           1: [0.001, 0.01]}
                # self.rShootNotInShootRange = {0: [0.005, 0.05],
                #                               1: [0.0001, 0.001]}
                self.rShootNotInSightRange = {0: [0.001, 0.01],
                                              1: [0.001, 0.01]}
                self.rCameraRange = {0: [0.001, 0.01],
                                     1: [0.001, 0.01]}
                self.speed = kwargs.get("speed")
                self.last_t = 0

            def reset(self, policy_modes):
                self._set_hit_rate()
                self._set_random_square_targets()
                self._set_random_circle_targets()
                self._get_rewards(policy_modes)
                self._reset_count()

                reset_info = super().reset()
                restartGame = reset_info["restartGame"]

                observation = restartGame["observation"]
                player = restartGame["player"]

                self._process_reset_observation(observation)
                self._process_reset_player(player, policy_modes)

                self._prepare_info(observation, player, None)
                a = np.array([[0] * 4] * self.agent_num)
                self.action_dict = self._get_action_dict(a)
                x_target_prop = np.random.uniform(0.3, 0.7, (self.agent_num, 1))
                y_target_prop = np.random.uniform(0.3, 0.7, (self.agent_num, 1))
                x_target = x_target_prop * self.x_range + self.x_min
                y_target = y_target_prop * self.y_range + self.y_min
                player_target = np.concatenate([x_target, y_target], axis=-1)
                target_dists = np.sum((self.player_location - player_target) ** 2, axis=-1) ** 0.5
                self.player_target = [(np.array(_player_target), target_dist)
                                      for _player_target, target_dist in zip(player_target, target_dists)]

                s, r, done, info = self.step(a)
                for i in range(self.agent_num):
                    self.action_dict[i]["moveX"] = self.player_location[i][0]
                    self.action_dict[i]["moveY"] = self.player_location[i][1]
                    self.action_dict[i]["moveD"] = self.x_range + self.y_range

                self._prepare_info(observation, player, self.action_dict)
                self.last_t = time.time()

                return s

            def step(self, a):
                action_dict = self._get_action_dict(a)

                s, r, done, info = super().step(action_dict)

                step = s["step"]
                observation = step["observation"]
                player = step["player"]

                t = time.time()
                interval_time = (t - self.last_t) * self.speed
                self.last_t = t
                embedding_t = np.array(
                    _power_embed(_clip(interval_time * 1000, 0.0, 1000.0), 0.0, 1000.0, 8, 0.75, True), np.float32)

                alphas = self._process_step_alphas()
                # start = time.time()
                rewards = self._process_step_reward(observation, player, action_dict, alphas)
                # if alphas is None:
                #     alphas = np.ones((len(player), 2), np.float32) / 2.0
                r = np.array(
                    [(alphas_i[0] * rewards_i["move"] + alphas_i[1] * rewards_i["fight"])
                     for rewards_i, alphas_i in zip(rewards, alphas)], np.float32)
                # print("process reward time {:.2f}ms".format(1000*(time.time()-start)))

                self._process_step_observation(observation)
                states = self._process_step_player(player, action_dict)
                s = []
                for s_i, r_i, al_i in zip(states, rewards, alphas):
                    ss = s_i
                    ss.update(r_i)
                    ss.update({'alphas': al_i})
                    ss.update({'interval_time': embedding_t})
                    s.append(ss)
                self._prepare_info(observation, player, action_dict)
                done = done["all"]

                info = (np.array([self.shoot_count, self.hit_count, self.in_poison_count, self.arrive_tar_count,
                        self.change_tar_count, self.face_move_count, self.move_count, self.find_enemy_count], dtype=int), self.reward_dict)

                return s, r, done, info

            def _reset_count(self):
                self.shoot_count = [0 for _ in range(self.agent_num)]
                self.hit_count = [0 for _ in range(self.agent_num)]
                self.in_poison_count = [0 for _ in range(self.agent_num)]
                self.arrive_tar_count = [0 for _ in range(self.agent_num)]
                self.change_tar_count = [0 for _ in range(self.agent_num)]
                self.find_enemy_count = [0 for _ in range(self.agent_num)]
                self.face_move_count = [0 for _ in range(self.agent_num)]
                self.move_count = [0 for _ in range(self.agent_num)]
                self.reward_dict = {"be_hurt": np.zeros(self.agent_num, np.float32),
                                    "itr_move": np.zeros(self.agent_num, np.float32),
                                    "spd_move": np.zeros(self.agent_num, np.float32),
                                    "cts_move": np.zeros(self.agent_num, np.float32),
                                    "face_move": np.zeros(self.agent_num, np.float32),
                                    "tar_move": np.zeros(self.agent_num, np.float32),
                                    "shoot": np.zeros(self.agent_num, np.float32),
                                    "sight_shoot": np.zeros(self.agent_num, np.float32),
                                    "no_shoot_fight": np.zeros(self.agent_num, np.float32),
                                    "no_shoot_move": np.zeros(self.agent_num, np.float32),
                                    "camera_fight": np.zeros(self.agent_num, np.float32),
                                    "camera_move": np.zeros(self.agent_num, np.float32)}

            def _set_hit_rate(self):
                self.hit_rate = self._log_uniform(
                    self.hit_rate_range, [0] * self.agent_num)

            def _set_random_square_targets(self):
                self._square_targets = np.random.uniform(0.3, 0.7, (8, 2))

            def _set_random_circle_targets(self):
                circle_radiuss = np.random.uniform(0.1, 0.9, (8, 1))
                circle_thetas = np.random.uniform(0.0, np.pi * 2, (8, 1))
                self._circle_targets = np.concatenate((circle_radiuss, circle_thetas), axis=-1)

            def _get_rewards(self, policy_modes):
                self.trace_memory = self._log_uniform(self.trace_memory_range, [0] * self.agent_num)
                self.trace_length = self._log_uniform(self.trace_length_range, [0] * self.agent_num)
                self.trace_rate = np.exp(np.log(self.trace_memory) / self.trace_length)

                self.rBuff, self.rBuff_prop = self._log_uniform(
                    self.rBuffRange, policy_modes, return_prop=True)
                self.rBox, self.rBox_prop = self._log_uniform(
                    self.rBoxRange, policy_modes, return_prop=True)
                self.rGetCrystal, self.rGetCrystal_prop = self._log_uniform(
                    self.rGetCrystalRange, policy_modes, return_prop=True)
                self.rLoseCrystal, self.rLoseCrystal_prop = self._log_uniform(
                    self.rLoseCrystalRange, policy_modes, return_prop=True)
                self.rKill, self.rKill_prop = self._log_uniform(
                    self.rKillRange, policy_modes, return_prop=True)
                self.rDeath, self.rDeath_prop = self._log_uniform(
                    self.rDeathRange, policy_modes, return_prop=True)
                self.rHurt, self.rHurt_prop = self._log_uniform(
                    self.rHurtRange, policy_modes, return_prop=True)
                self.rBeHurt, self.rBeHurt_prop = self._log_uniform(
                    self.rBeHurtRange, policy_modes, return_prop=True)
                self.rInterruptMove, self.rInterruptMove_prop = self._log_uniform(
                    self.rInterruptMoveRange, policy_modes, return_prop=True)
                self.rFaceMove, self.rFaceMove_prop = self._log_uniform(
                    self.rFaceMoveRange, policy_modes, return_prop=True)
                self.rContinuousMove, self.rContinuousMove_prop = self._log_uniform(
                    self.rContinuousMoveRange, policy_modes, return_prop=True)
                self.rShootInShoot, self.rShootInShoot_prop = self._log_uniform(
                    self.rShootInShootRange, policy_modes, return_prop=True)
                self.rShootInSight, self.rShootInSight_prop = self._log_uniform(
                    self.rShootInSightRange, policy_modes, return_prop=True)
                self.rShootNotInSight, self.rShootNotInSight_prop = self._log_uniform(
                    self.rShootNotInSightRange, policy_modes, return_prop=True)
                self.rCamera, self.rCamera_prop = self._log_uniform(
                    self.rCameraRange, policy_modes, return_prop=True)

            def _get_action_dict(self, a):
                action_dict = dict()
                player_distance = getattr(self, "player_distance", None)
                angle = getattr(self, "angle", None)
                shoot_index = getattr(self, "shoot_index", None)
                for i, a_i, in enumerate(a):
                    d = dict()
                    d["jump"] = 0
                    d["squat"] = 0
                    d["reload"] = 0
                    d["switch"] = 0
                    d["cure"] = 0
                    d["timeSpan"] = 0
                    last_action_dict = getattr(self, "action_dict", None)
                    if a_i[0] == 1:
                        rotation = self.player_rotation[i]
                        x, y = self.player_location[i]
                        r_i, theta_i = divmod(a_i[1], 12)
                        r = [7.5, 22.5][r_i]
                        theta = theta_i * np.pi / 6.0 + rotation
                        d["moveX"] = x + r * math.sin(theta)
                        d["moveY"] = y + r * math.cos(theta)
                        d["moveD"] = r
                        self.move_count[i] += 1
                    else:
                        if last_action_dict is not None:
                            d["moveX"] = self.action_dict[i]["moveX"]
                            d["moveY"] = self.action_dict[i]["moveY"]
                            d["moveD"] = self.action_dict[i]["moveD"]
                        else:
                            d["moveX"], d["moveY"] = self.player_location[i]
                            d["moveD"] = self.x_range + self.y_range
                    if a_i[2] == 0:
                        d["camera"] = a_i[2]
                        if np.random.random() < 0.01:
                            if np.random.random() < 0.5:
                                d["camera"] = 8
                            else:
                                d["camera"] = 9
                    else:
                        d["camera"] = a_i[2] + 3
                        if np.random.random() < 0.01:
                            if np.random.random() < 0.5:
                                d["camera"] += 1
                            else:
                                d["camera"] -= 1
                    if a_i[3] == 1:
                        hit_rate = self.hit_rate[i]
                        if player_distance is not None and angle is not None and shoot_index is not None:
                            if shoot_index[i] != -1:
                                hit_rate *= hit_rate_func(
                                    player_distance[i][shoot_index[i]], angle[i][shoot_index[i]])
                            else:
                                hit_rate = 0.0
                        if np.random.random() < hit_rate:
                            d["shoot"] = 1
                            self.shoot_count[i] += 1
                            if a_i[0] == 0:
                                if np.random.random() < 0.01:
                                    r = np.random.uniform(0, 5)
                                    theta = np.random.uniform(0, np.pi * 2)
                                    d["moveX"] += r * math.sin(theta)
                                    d["moveY"] += r * math.cos(theta)
                        else:
                            d["shoot"] = 0
                    else:
                        d["shoot"] = 0
                    action_dict[i] = d

                    # if a_i[0] == 1:
                    #     rel_pos_x, rel_pos_y = d["moveX"] - self.player_location[i][0], d["moveY"] - self.player_location[i][1]
                    #     rel_angle = np.arctan2(rel_pos_x, rel_pos_y)
                    #     rel_angle %= (2 * np.pi)
                    #     rota_angle = self.player_rotation[i]
                    #     # calculate the angle of view
                    #     camera_action = np.array(
                    #         [0, -60, -52.5, -45, -37.5, -30, -22.5, -15, -7.5, 7.5, 15, 22.5, 30, 37.5, 45, 52.5, 60])
                    #     after_action = ((camera_action / 180.0 * np.pi) + rota_angle) % (2 * np.pi)
                    #     after_action = [np.min([np.abs(rel_angle - aa), (2 * np.pi) - np.abs(rel_angle - aa)]) for aa in
                    #                     after_action]
                    #     if d["camera"] - 1 <= np.argmin(after_action) <= d["camera"] + 1:
                    #         self.face_move_count[i] += 1

                return action_dict

            def _process_reset_observation(self, observation):
                self.gameId = observation["gameId"]

                '''
                map_info
                '''
                self.x_min = int(observation["xMin"])
                self.x_max = int(observation["xMax"])
                self.y_min = int(observation["yMin"])
                self.y_max = int(observation["yMax"])
                self.x_range = self.x_max - self.x_min + 1
                self.y_range = self.y_max - self.y_min + 1

                '''
                process camera vision to (r, theta) coordinates
                '''
                self.cameraWidth = observation["cameraWidth"] / 180.0 * np.pi
                self.cameraRange = observation["cameraRange"]
                self.polar_r = np.linspace(0, self.cameraRange, shape[0] + 1)[1:]
                self.polar_theta = np.linspace(-self.cameraWidth / 2, self.cameraWidth / 2, shape[1])
                self.voice_r = np.array([20 - 20 * (1 - k / 10) ** 0.5 for k in range(1, 11)])
                self.voice_theta = np.linspace(self.cameraWidth / 2, 2 * np.pi - self.cameraWidth / 2, 2 * shape[1])
                '''
                process mapInfo
                '''
                self.mapHeight = (np.reshape(
                    observation["mapHeight"], (self.x_range, self.y_range)
                ) - observation["seaHorizonHeight"]) / 100.0
                self.mapReach = np.reshape(
                    observation["mapReach"], (self.x_range, self.y_range))
                mapLandscape = observation["mapLandscape"]
                self.mapLandscape = np.reshape(
                    mapLandscape, (self.x_range, self.y_range))

                n_values = landscape_n_values
                mapLandscape_onehot = np.reshape(
                    np.eye(n_values)[mapLandscape],
                    (self.x_range, self.y_range, n_values))
                self.mapInfo = np.concatenate(
                    [mapLandscape_onehot,
                     self.mapHeight[:, :, None],
                     self.mapReach[:, :, None]],
                    axis=-1)

                '''
                add 7 channels for loc, #box, #buff, #bullets, crystal, #weapon, poison
                '''
                self.mapInfo = np.concatenate(
                    [np.zeros((self.x_range, self.y_range, 3)),
                     self.mapInfo],
                    axis=-1)

                """
                minimap for poison, all player loc, player loc, target loc
                """
                self.mini_map = np.zeros(shape=(shape[0], shape[1], 4), dtype=np.float32)

            def _process_reset_player(self, player, policy_modes):
                n_players = len(player)
                if n_players > 15:
                    mode = 1.0
                else:
                    mode = 0.0
                self.playerStatic = []
                self.playerStaticVec = []
                self.playerMeta = []
                self.playerXtrace = np.zeros(self.agent_num)
                self.playerYtrace = np.zeros(self.agent_num)
                self.id2idx = dict()
                for i, player_i in enumerate(player):
                    keys = ["playerId", "playerRole",
                            "weaponDamage", "weaponRange",
                            "weaponSpeed", "weaponBullets",
                            "fullHp", "fullShield", "fullHpPacks"]
                    self.playerStatic.append(
                        {key: player_i[key] for key in keys})
                    self.id2idx[player_i["playerId"]] = i

                    player_i_vec = [
                        player_i["weaponDamage"] / 50.0,
                        player_i["weaponRange"] / self.cameraRange,
                        player_i["weaponSpeed"] / 10.0,
                        player_i["weaponBullets"] / 50.0,
                        mode,
                    ]
                    self.playerStaticVec.append(player_i_vec)

                    player_i_meta = [
                        self._log_uniform_proportion(
                            self.trace_memory_range, self.trace_memory[i], 0),
                        self._log_uniform_proportion(
                            self.trace_length_range, self.trace_length[i], 0),
                        self._log_uniform_proportion(
                            self.rBuffRange, self.rBuff[i], policy_modes[i]),
                        self._log_uniform_proportion(
                            self.rBoxRange, self.rBox[i], policy_modes[i]),
                        # self._log_uniform_proportion(
                        #     self.rBulletRange, self.rBullet[i], policy_modes[i]),
                        # self._log_uniform_proportion(
                        #     self.rWeaponRange, self.rWeapon[i], policy_modes[i]),
                        self._log_uniform_proportion(
                            self.rGetCrystalRange, self.rGetCrystal[i], policy_modes[i]),
                        self._log_uniform_proportion(
                            self.rLoseCrystalRange, self.rLoseCrystal[i], policy_modes[i]),
                        self._log_uniform_proportion(
                            self.rKillRange, self.rKill[i], policy_modes[i]),
                        self._log_uniform_proportion(
                            self.rDeathRange, self.rDeath[i], policy_modes[i]),
                        # self._log_uniform_proportion(
                        #     self.rHpRange, self.rHp[i], policy_modes[i]),
                        # self._log_uniform_proportion(
                        #     self.rShieldRange, self.rShield[i], policy_modes[i]),
                        self._log_uniform_proportion(
                            self.rHurtRange, self.rHurt[i], policy_modes[i]),
                        self._log_uniform_proportion(
                            self.rBeHurtRange, self.rBeHurt[i], policy_modes[i]),
                        self._log_uniform_proportion(
                            self.rInterruptMoveRange, self.rInterruptMove[i], policy_modes[i]),
                        self._log_uniform_proportion(
                            self.rFaceMoveRange, self.rFaceMove[i], policy_modes[i]),
                        self._log_uniform_proportion(
                            self.rContinuousMoveRange, self.rContinuousMove[i], policy_modes[i]),
                        # self._log_uniform_proportion(
                        #     self.rFinishMoveRange, self.rFinishMove[i], policy_modes[i]),
                        self._log_uniform_proportion(
                            self.rShootInShootRange, self.rShootInShoot[i], policy_modes[i]),
                        self._log_uniform_proportion(
                            self.rShootInSightRange, self.rShootInSight[i], policy_modes[i]),
                        # self._log_uniform_proportion(
                        #     self.rShootNotInShootRange, self.rShootNotInShoot[i], policy_modes[i]),
                        self._log_uniform_proportion(
                            self.rShootNotInSightRange, self.rShootNotInSight[i], policy_modes[i]),
                        self._log_uniform_proportion(
                            self.rCameraRange, self.rCamera[i], policy_modes[i]),
                    ]
                    player_i_meta = [j for i in player_i_meta for j in i]
                    self.playerMeta.append(player_i_meta)

            def _process_step_observation(self, observation):
                mapBox = self._get_locations(
                    observation.get("mapBox", None))
                mapBuff = self._get_locations(
                    observation.get("mapBuff", None))
                # mapBullets = self._get_locations(
                #     observation.get("mapBullets", None))
                mapCrystal = self._get_locations(
                    observation.get("mapCrystal", None))
                # mapWeapon = self._get_locations(
                #     observation.get("mapWeapon", None))

                prev_mapBox = getattr(self, "mapBox", [])
                self.mapInfo = self._register_location(
                    self.mapInfo, 2, mapBox, prev_mapBox)

                prev_mapBuff = getattr(self, "mapBuff", [])
                self.mapInfo = self._register_location(
                    self.mapInfo, 2, mapBuff, prev_mapBuff)

                prev_mapCrystal = getattr(self, "mapCrystal", [])
                self.mapInfo = self._register_location(
                    self.mapInfo, 2, mapCrystal, prev_mapCrystal)

                # prev_mapBullets = getattr(self, "mapBullets", [])
                # self.mapInfo = self._register_location(
                #     self.mapInfo, 5, mapBullets, prev_mapBullets)

                # prev_mapWeapon = getattr(self, "mapWeapon", [])
                # self.mapInfo = self._register_location(
                #     self.mapInfo, 6, mapWeapon, prev_mapWeapon)

                has_poison = observation.get("hasPoison", False)
                poison_radius = observation.get("poisonRadius", 999999.0)
                poison_center_x = observation.get("poisonCenterX", 0.0)
                poison_center_y = observation.get("poisonCenterY", 0.0)
                prev_has_poison = self.observation.get("hasPoison", False)
                prev_poison_radius = self.observation.get("poisonRadius", 999999.0)
                prev_poison_center_x = self.observation.get("poisonCenterX", 0.0)
                prev_poison_center_y = self.observation.get("poisonCenterY", 0.0)

                if (has_poison != prev_has_poison) or (
                        poison_radius != prev_poison_radius) or (
                        poison_center_x != prev_poison_center_x) or (
                        poison_center_y != prev_poison_center_y):
                    if not has_poison:
                        self.mapInfo[:, :, 1] = np.zeros_like(self.mapInfo[:, :, 1])
                        self.mini_map[:, :, 1] = np.zeros(shape=shape, dtype=np.float32)
                    else:
                        x_idxes = np.arange(self.x_min, self.x_max + 1)
                        y_idxes = np.arange(self.y_min, self.y_max + 1)
                        x_dists = x_idxes - poison_center_x
                        y_dists = y_idxes - poison_center_y
                        dists = (x_dists[:, None] ** 2 + y_dists[None, :] ** 2) ** 0.5
                        self.mapInfo[:, :, 1] = np.array(dists > poison_radius, dtype=np.float32)

                        x_idxes = np.arange(0, shape[0])
                        y_idxes = np.arange(0, shape[1])
                        x_dists = x_idxes / (shape[0] - 1) * self.x_range + self.x_min - poison_center_x
                        y_dists = y_idxes / (shape[1] - 1) * self.y_range + self.y_min - poison_center_y
                        dists = (x_dists[:, None] ** 2 + y_dists[None, :] ** 2) ** 0.5
                        self.mini_map[:, :, 1] = np.array(dists > poison_radius, dtype=np.float32)

            def _register_minimap(self, locations, channel):
                self.mini_map[:, :, channel] = np.zeros_like(self.mini_map[:, :, channel])
                for loc in locations:
                    x, y = loc
                    x = (x - self.x_min) / self.x_range * (shape[0] - 1)
                    y = (y - self.y_min) / self.y_range * (shape[1] - 1)
                    x_l, x_r = math.floor(x), math.ceil(x)
                    y_l, y_r = math.floor(y), math.ceil(y)
                    x_lw, x_rw = x_r - x, x - x_l
                    y_lw, y_rw = y_r - y, y - y_l
                    if x_l == x_r:
                        x_lw += 1.0
                    if y_l == y_r:
                        y_lw += 1.0
                    try:
                        self.mini_map[x_l, y_l, channel] += x_lw * y_lw
                        self.mini_map[x_l, y_r, channel] += x_lw * y_rw
                        self.mini_map[x_r, y_l, channel] += x_rw * y_lw
                        self.mini_map[x_r, y_r, channel] += x_rw * y_rw
                    except Exception as e:
                        print(e)
                        print(locations)
                        print(loc, self.x_min, self.x_max, self.y_min, self.y_max)

            def _process_step_player(self, player, action_dict):
                prev_player_location = getattr(self, "player_location", [])
                player_location = [[player_i["x"], player_i["y"]] for player_i in player]
                self.mapInfo = self._register_location(
                    self.mapInfo, 0, player_location, prev_player_location)
                self._register_minimap(player_location, 0)

                player_location = np.array(player_location)
                rotations = np.array(
                    [player_i["rotation"] / 180.0 * np.pi for player_i in player])

                self.player_distance, self.angle, self.shoot_index = enemy_location(
                    player_location, rotations, 50, np.pi / 6)

                player_distance = self.player_distance + 65535 * np.eye(self.agent_num)
                nearest = np.argmin(player_distance, axis=-1)

                has_poison = self.observation.get("hasPoison", 0.0)

                playerState = []

                '''
                get player image info
                '''
                xs = np.array([player_i["x"] for player_i in player])
                ys = np.array([player_i["y"] for player_i in player])
                camera_theta = self.polar_theta[None, None, :] + rotations[:, None, None]
                camera_xs = (self.polar_r[None, :, None] * np.sin(camera_theta)
                             + xs[:, None, None] - self.x_min)
                camera_ys = (self.polar_r[None, :, None] * np.cos(camera_theta)
                             + ys[:, None, None] - self.y_min)
                voice_theta = self.voice_theta[None, None, :] + rotations[:, None, None]
                voice_xs = (self.voice_r[None, :, None] * np.sin(voice_theta)
                            + xs[:, None, None] - self.x_min)
                voice_ys = (self.voice_r[None, :, None] * np.cos(voice_theta)
                            + ys[:, None, None] - self.y_min)

                # xs = np.array(np.round(camera_xs), np.int32)
                # ys = np.array(np.round(camera_ys), np.int32)
                # xs = np.minimum(np.maximum(xs, 0), self.x_range - 1)
                # ys = np.minimum(np.maximum(ys, 0), self.y_range - 1)

                camera_xs = np.minimum(np.maximum(camera_xs, 0), self.x_range - 1)
                camera_ys = np.minimum(np.maximum(camera_ys, 0), self.y_range - 1)
                voice_xs = np.minimum(np.maximum(voice_xs, 0), self.x_range - 1)
                voice_ys = np.minimum(np.maximum(voice_ys, 0), self.y_range - 1)

                for i in range(len(player)):
                    player_i = player[i]
                    static_i = self.playerStatic[i]
                    assert player_i["playerId"] == static_i["playerId"]

                    '''
                    get player vector info
                    '''
                    info = copy.deepcopy(self.playerStaticVec[i])
                    info.append(has_poison)
                    info.append(player_i["aliveOrDead"])
                    info.append(player_i["standOrSquat"])
                    info.append(player_i["hpPacks"] / 5.0)
                    info.append(player_i["enemyInSight"])
                    info.append(player_i["enemyInShoot"])
                    info.append(player_i["zombieInSight"])
                    info.append(player_i["zombieInShoot"])

                    player_i["hp"] = _clip(player_i["hp"], 0.0, static_i["fullHp"])
                    info.extend(_sqrt_embed(player_i["hp"], 0.0, static_i["fullHp"], 8))

                    player_i["shield"] = _clip(player_i["shield"], 0.0, static_i["fullShield"])
                    info.extend(_sqrt_embed(player_i["shield"], 0.0, static_i["fullShield"], 4))

                    info.append(player_i["bullets"] / 50.0)
                    info.append(player_i["kill"] / 5.0)
                    info.append(player_i["death"] / 5.0)
                    info.extend(_power_embed(_clip(player_i["crystal"], 0.0, 200.0), 0.0, 200.0, 5, 0.75))

                    num_weapon = int(player_i["hasWeapon"])
                    has_weapon = 1.0 * (num_weapon > 0)
                    info.append(has_weapon)
                    # weapon_onehot = [1.0] * num_weapon + [0.0] * (3 - num_weapon)
                    # info += weapon_onehot

                    x = player_i["x"]
                    y = player_i["y"]
                    self._register_minimap([(x, y)], 2)
                    x_proportion = (x - self.x_min) / self.x_range
                    y_proportion = (y - self.y_min) / self.y_range
                    info.extend(_binary_embed(x_proportion, 0.0, 1.0, 8))
                    info.extend(_binary_embed(y_proportion, 0.0, 1.0, 8))

                    sin, cos = math.sin(rotations[i]), math.cos(rotations[i])
                    info.extend(_soft_rot_embed(cos, sin, 4, 0.0))

                    target_x, target_y = self.player_target[i][0]
                    target_x, target_y = float(target_x), float(target_y)
                    self._register_minimap([(target_x, target_y)], 3)
                    info.extend(_soft_rot_embed(target_y - y, target_x - x, 8, rotations[i]))

                    target_distance = ((target_x - x) ** 2 + (target_y - y) ** 2) ** 0.5
                    proportion = target_distance / ((self.x_range + self.y_range) / 2)
                    proportion = _clip(proportion, 0.0, 1.0)
                    info.extend(_binary_embed(proportion, 0.0, 1.0, 4, 0.5))

                    nearest_x, nearest_y = player_location[nearest[i]]
                    nearest_x, nearest_y = float(nearest_x), float(nearest_y)
                    info.extend(_soft_rot_embed(nearest_y - y, nearest_x - x, 4, rotations[i]))
                    nearest_dist = ((nearest_x - x) ** 2 + (nearest_y - y) ** 2) ** 0.5
                    info.extend(_power_embed(_clip(
                        nearest_dist, 0.0, self.cameraRange), 0.0, self.cameraRange, 3, 1.0))

                    moveX, moveY = action_dict[i]["moveX"], action_dict[i]["moveY"]
                    info.extend(_soft_rot_embed(moveY - y, moveX - x, 4, rotations[i]))

                    distance = ((moveX - x) ** 2 + (moveY - y) ** 2) ** 0.5
                    moveD = action_dict[i]["moveD"]
                    unfinish = min(distance / max(moveD, 1e-8), 1.0)
                    info.extend(_power_embed(unfinish, 0.0, 1.0, 3, 1.0))

                    hurt = player_i["hurt"]
                    hurtX = player_i["hurtX"]
                    hurtY = player_i["hurtY"]
                    info.append(hurt)
                    if hurt:
                        relative_theta_info = _soft_rot_embed(hurtY - y, hurtX - x, 4, rotations[i])
                    else:
                        relative_theta_info = [0.0] * 4
                    info += relative_theta_info

                    playerStepVec = info

                    '''
                    get player image info
                    '''
                    image = self._bilinear_approx(camera_xs[i], camera_ys[i], self.mapInfo)
                    voice = self._bilinear_approx(voice_xs[i], voice_ys[i], self.mapInfo[:, :, 0:1])
                    self.find_enemy_count[i] += (np.sum(image[1:-1, 1:-1, 0]) >= 1)
                    playerStepImage = np.array(100 * np.concatenate(
                        [image, self.mini_map], axis=-1), dtype=np.uint8)
                    playerVoice = np.array(100 * voice, dtype=np.uint8)

                    playerState.append({"image": playerStepImage,
                                        "vec": np.array(playerStepVec, np.float32),
                                        "meta": np.array(self.playerMeta[i], np.float32),
                                        "voice": playerVoice})
                return playerState

            @staticmethod
            def _bilinear_approx(xs, ys, mapInfo):
                x_l, x_r = np.array(np.floor(xs), np.int32), np.array(np.ceil(xs), np.int32)
                y_l, y_r = np.array(np.floor(ys), np.int32), np.array(np.ceil(ys), np.int32)
                x_lw, x_rw = x_r - xs, xs - x_l
                y_lw, y_rw = y_r - ys, ys - y_l
                x_lw += (x_l == x_r)
                y_lw += (y_l == y_r)
                image = (
                        mapInfo[[x_l, y_l]] * x_lw[:, :, None] * y_lw[:, :, None]
                        + mapInfo[[x_r, y_l]] * x_rw[:, :, None] * y_lw[:, :, None]
                        + mapInfo[[x_l, y_r]] * x_lw[:, :, None] * y_rw[:, :, None]
                        + mapInfo[[x_r, y_r]] * x_rw[:, :, None] * y_rw[:, :, None]
                )
                return image

            def _process_step_alphas(self):
                player_distance = getattr(self, "player_distance", None)
                if player_distance is not None:
                    player_distance = self.player_distance + 65535 * np.eye(self.agent_num)
                    min_distance = np.min(player_distance, axis=-1)
                    alphas = np.where(min_distance < 50, 0, 1)
                    # alphas: [move, fight]
                    alphas = np.array(np.transpose([alphas, 1 - alphas]), dtype=np.float32)
                else:
                    alphas = np.ones((self.agent_num, 2), np.float32) / 2.0
                return alphas

            def _process_step_reward(self, observation, player, action_dict, alphas):
                n_players = len(player)
                assert n_players == self.agent_num
                rewards = {"move": np.zeros(n_players, np.float32),
                           "fight": np.zeros(n_players, np.float32), }


                '''
                task:
                    0 : move
                    # 1 : pick
                    2 : fight
                    # 3 : escape
                '''

                '''
                get new player information & update r
                '''
                playerKill = np.array([player_i["kill"] for player_i in player])
                playerDeath = np.array([player_i["death"] for player_i in player])
                playerCrystal = np.array([player_i["crystal"] for player_i in player])

                tmp = self.rKill * (playerKill - self.playerKill)
                rewards["fight"] += tmp * 2.142857
                # reward_dict["kill"] += tmp * 2.142857

                tmp = self.rDeath * (playerDeath - self.playerDeath)
                rewards["fight"] -= tmp
                # reward_dict["death"] -= tmp
                # rewards["move"] -= tmp

                # tmp = self.rGetCrystal * np.maximum(
                #     playerCrystal - self.playerCrystal, np.zeros(n_players))
                # rewards["fight"] += tmp * 0.428571
                # rewards["move"] += tmp * 2.142857
                #
                # tmp = self.rLoseCrystal * np.maximum(
                #     self.playerCrystal - playerCrystal, np.zeros(n_players))
                # rewards["fight"] -= tmp
                # rewards["move"] -= tmp

                hurt = np.array([player_i["hurt"] for player_i in player])
                hurtId = np.array([player_i["hurtId"] for player_i in player])
                for i, hurt_i in enumerate(hurt):
                    if hurt_i:
                        tmp = self.rBeHurt[i]
                        rewards["fight"][i] -= tmp * 1.571428
                        self.reward_dict["be_hurt"][i] -= tmp * alphas[i][1]
                        # rewards["move"][i] -= tmp * 0.714285
                        # rewards["pick"][i] -= tmp * 0.714285
                        # r[self.id2idx[hurtId[i]]] += self.rHp[self.id2idx[hurtId[i]]] * hp_loss
                        # r[self.id2idx[hurtId[i]]] += self.rShield[self.id2idx[hurtId[i]]] * shield_loss
                        tmp = self.rHurt[self.id2idx[hurtId[i]]]
                        rewards["fight"][self.id2idx[hurtId[i]]] += tmp * 2.142857
                        # rewards["move"][self.id2idx[hurtId[i]]] += tmp * 0.428571
                        # rewards["pick"][self.id2idx[hurtId[i]]] += tmp * 0.428571
                        self.hit_count[self.id2idx[hurtId[i]]] += 1
                        # reward_dict["hurt"][self.id2idx[hurtId[i]]] += tmp * 2.142857


                # '''
                # get new items locations & update r
                # '''
                # mapBox = self._get_locations(
                #     observation.get("mapBox", None))
                # mapBuff = self._get_locations(
                #     observation.get("mapBuff", None))
                # # mapBullets = self._get_locations(
                # #     observation.get("mapBullets", None))
                # # mapWeapon = self._get_locations(
                # #     observation.get("mapWeapon", None))
                #
                # pick_box = []
                # for box in self.mapBox:
                #     if box not in mapBox:
                #         pick_box.append(box)
                # pick_box = np.reshape(pick_box, (-1, 2))
                # distance = np.sum(np.square(
                #     pick_box[:, None, :] - self.player_location[None, :, :]),
                #     axis=-1)
                # pick = np.sum(distance < 16, axis=0)
                # tmp = self.rBox * pick
                # # rewards["fight"] += tmp * 0.428571
                # # rewards["move"] += tmp * 0.428571
                # rewards["move"] += tmp * 2.142857
                #
                # pick_buff = []
                # for buff in self.mapBuff:
                #     if buff not in mapBuff:
                #         pick_buff.append(buff)
                # pick_buff = np.reshape(pick_buff, (-1, 2))
                # distance = np.sum(np.square(
                #     pick_buff[:, None, :] - self.player_location[None, :, :]),
                #     axis=-1)
                # pick = np.sum(distance < 16, axis=0)
                # tmp = self.rBuff * pick
                # # rewards["fight"] += tmp * 0.428571
                # # rewards["move"] += tmp * 0.428571
                # rewards["move"] += tmp * 2.142857

                '''
                get moveX moveY & update r
                '''
                for i in range(n_players):
                    x, y = self.player_location[i]
                    rotation = player[i]["rotation"] / 180.0 * np.pi
                    # InterruptMove
                    if (action_dict[i]["moveX"] != self.action_dict[i]["moveX"]) and (
                            action_dict[i]["moveY"] != self.action_dict[i]["moveY"]):
                        moveX = self.action_dict[i]["moveX"]
                        moveY = self.action_dict[i]["moveY"]
                        moveD = self.action_dict[i]["moveD"]
                        if ((moveX - x) ** 2 + (moveY - y) ** 2) ** 0.5 > 0.5 * moveD:
                            # rewards["fight"][i] += 0.1 * self.rInterruptMove[i]
                            rewards["move"][i] -= self.rInterruptMove[i]
                            # rewards["pick"][i] -= self.rInterruptMove[i]
                            self.reward_dict["itr_move"][i] -= self.rInterruptMove[i] * alphas[i][0]

                    move_distance = ((player[i]["x"] - x) ** 2 + (player[i]["y"] - y) ** 2) ** 0.5
                    if move_distance < 0.2:
                        rewards["move"][i] -= 0.5
                        # rewards["pick"][i] += 0.01
                        self.reward_dict["spd_move"][i] -= 0.5 * alphas[i][0]

                    unit_x, unit_y = (player[i]["x"] - x) / 12.5, (player[i]["y"] - y) / 12.5
                    unit_dist = (unit_x ** 2 + unit_y ** 2) ** 0.5
                    delta_x = unit_x * min(1, 1 / unit_dist)  # * min(1.0 / max(move_distance, 1e-8), 1.0)
                    delta_y = unit_y * min(1, 1 / unit_dist)  # * min(1.0 / max(move_distance, 1e-8), 1.0)

                    '''
                    cts move & face move
                    '''
                    cosine = (delta_x * self.playerXtrace[i] + delta_y * self.playerYtrace[i])
                    # rewards["fight"] += self.rContinuousMove[i] * (99.0 + cosine) / 100.0 * 0.428571
                    rewards["move"][i] += self.rContinuousMove[i] * _rescale((0.25 + cosine) / 1.25) * 2.142857
                    # rewards["pick"] += self.rContinuousMove[i] * (2.5 + cosine) / 3.5 * 0.428571
                    self.reward_dict["cts_move"][i] += self.rContinuousMove[i] * _rescale((0.25 + cosine) / 1.25) * 2.142857 * alphas[i][0]

                    self.playerXtrace[i] = delta_x + self.trace_rate[i] * self.playerXtrace[i]
                    self.playerYtrace[i] = delta_y + self.trace_rate[i] * self.playerYtrace[i]
                    cosine = (np.sin(rotation) * self.playerXtrace[i]
                              + np.cos(rotation) * self.playerYtrace[i])
                    if cosine > 0:
                        self.face_move_count[i] += 1
                    # rewards["fight"] += self.rFaceMove[i] * (1.5 - np.abs(cosine)) / 1.5 * 0.428571
                    rewards["move"][i] += self.rFaceMove[i] * _rescale((0.0 + cosine) / 1.0) * 2.142857
                    # rewards["pick"] += self.rFaceMove[i] * _rescale((0.5 + cosine) / 1.5) * 0.428571
                    self.reward_dict["face_move"][i] += self.rFaceMove[i] * _rescale((0.0 + cosine) / 1.0) * 2.142857 * alphas[i][0]

                    '''
                    target move
                    '''
                    has_poison = observation.get("hasPoison", False)
                    poison_radius = observation.get("poisonRadius", 999999.0)
                    poison_center_x = observation.get("poisonCenterX", 0.0)
                    poison_center_y = observation.get("poisonCenterY", 0.0)
                    old_locations = np.array(self.player_location[i])
                    new_locations = np.array([player[i]["x"], player[i]["y"]])
                    old_distance = np.sum((self.player_target[i][0] - old_locations) ** 2) ** 0.5
                    new_distance = np.sum((self.player_target[i][0] - new_locations) ** 2) ** 0.5
                    distance_reduce = _clip(old_distance - new_distance, -2.5, 12.5)
                    norm_old_distance = max((1 - (old_distance / self.player_target[i][1])), 0.0) ** 0.15
                    out_of_poison = bool(
                        ((x - poison_center_x) ** 2 +
                         (y - poison_center_y) ** 2) ** 0.5 > poison_radius
                    ) and bool(
                        ((player[i]["x"] - poison_center_x) ** 2 +
                         (player[i]["y"] - poison_center_y) ** 2) ** 0.5 > poison_radius)
                    if out_of_poison:
                        tmp = distance_reduce * max(self.rContinuousMove[i],
                                                    self.rFaceMove[i]) * 20.0 * norm_old_distance
                    else:
                        tmp = distance_reduce * max(self.rContinuousMove[i], self.rFaceMove[i]) * 1.0
                        self.in_poison_count[i] += 1
                    # rewards["fight"] += tmp * 0.428571
                    rewards["move"][i] += tmp * 2.142857
                    # rewards["pick"] += tmp * 0.428571
                    self.reward_dict["tar_move"][i] += tmp * 2.142857 * alphas[i][0]

                    # in poison reward
                    # if has_poison and not out_of_poison:
                    #     if poison_radius < 100:
                    #         poison_radius = 100
                    #     norm_poison_radius = poison_radius / ((self.x_range ** 2 + self.y_range ** 2) ** 0.5)
                    #     rewards["move"][i] += 0.03 / norm_poison_radius
                    #     rewards["fight"][i] += 0.05 / norm_poison_radius

                    if new_distance < 10:
                        self.arrive_tar_count[i] += 1
                    # generate new target
                    if has_poison:
                        dist = np.sum((self.player_target[i][0] - np.array([
                            poison_center_x, poison_center_y])) ** 2) ** 0.5
                        not_safe_target = bool(np.array(dist > poison_radius))
                        if not_safe_target or new_distance < 10:
                            x_target, y_target = self._circle_sample(poison_center_x, poison_center_y, poison_radius)
                            target_dist = ((x_target - x) ** 2 + (y_target - y) ** 2) ** 0.5
                            self.player_target[i] = (np.array([x_target, y_target]), target_dist)
                            self.change_tar_count[i] += 1
                    else:
                        if new_distance < 10:
                            if np.random.random() < 0.5:
                                x_target, y_target = self._square_sample()
                                target_dist = ((x_target - x) ** 2 + (y_target - y) ** 2) ** 0.5
                                self.player_target[i] = (np.array([x_target, y_target]), target_dist)
                                self.change_tar_count[i] += 1
                            else:
                                j = np.random.randint(0, self.agent_num)
                                x_target = self.player_location[j][0]
                                y_target = self.player_location[j][1]
                                target_dist = ((x_target - x) ** 2 + (y_target - y) ** 2) ** 0.5
                                self.player_target[i] = (np.array([x_target, y_target]), target_dist)
                                self.change_tar_count[i] += 1
                '''
                whether shoot or not & update r
                '''
                for i in range(n_players):
                    if action_dict[i]["shoot"] == 1:
                        inShoot = self.player[i]["enemyInShoot"] or self.player[i]["zombieInShoot"]
                        inSight = self.player[i]["enemyInSight"] or self.player[i]["zombieInSight"]
                        if inShoot:
                            rewards["fight"][i] += self.rShootInShoot[i] * 2.142857
                            self.reward_dict["shoot"][i] += self.rShootInShoot[i] * 2.142857 * alphas[i][1]
                            # rewards["move"][i] += self.rShootInShoot[i] * 0.428571
                            # rewards["pick"][i] += self.rShootInShoot[i] * 0.428571
                        # else:
                        #     r[i] -= self.rShootNotInShoot[i]
                        elif inSight: # 开空枪的次数过多
                            rewards["fight"][i] += self.rShootInSight[i] * 1.071428
                            self.reward_dict["sight_shoot"][i] += self.rShootInShoot[i] * 2.142857 * alphas[i][1]
                            # rewards["move"][i] += self.rShootInSight[i] * 0.428571
                            # rewards["pick"][i] += self.rShootInSight[i] * 0.428571
                        else:
                            rewards["fight"][i] -= self.rShootNotInSight[i] * 2.142857
                            rewards["move"][i] -= self.rShootNotInSight[i] * 0.428571
                            # rewards["pick"][i] -= self.rShootNotInSight[i] * 0.428571
                            self.reward_dict["no_shoot_fight"][i] -= self.rShootNotInSight[i] * 2.142857 * alphas[i][1]
                            self.reward_dict["no_shoot_move"][i] -= self.rShootNotInSight[i] * 0.428571 * alphas[i][0]
                '''
                get camera & update r
                '''
                for i in range(n_players):
                    camera = action_dict[i]["camera"]
                    if camera != 0:
                        if camera > 8:
                            d = camera - 8
                        else:
                            d = 9 - camera
                        rewards["move"][i] -= self.rCamera[i] * np.sin(d * np.pi / 16) * 10
                        self.reward_dict["camera_move"][i] -= self.rCamera[i] * np.sin(d * np.pi / 16) * 10 * alphas[i][0]

                        rewards["fight"][i] -= self.rCamera[i] * np.sin(d * np.pi / 16) * 2.5
                        self.reward_dict["camera_fight"][i] -= self.rCamera[i] * np.sin(d * np.pi / 16) * 10 * alphas[i][1]

                        # if player[i]["hurt"]:
                        #     rewards["fight"][i] += self.rCamera[i]
                        #     rewards["move"][i] += self.rCamera[i]
                        #     # rewards["pick"][i] += self.rCamera[i]
                        # else:
                        #     rewards["move"][i] -= self.rCamera[i] * np.sin(d * np.pi / 16) * 10
                        #     # rewards["pick"][i] -= self.rCamera[i] * np.sin(d * np.pi / 16)
                rewards = [{k: v[i] for k, v in rewards.items()} for i in range(n_players)]
                return rewards

            def _square_sample(self):
                idx = np.random.choice(np.arange(len(self._square_targets)))
                x_target_prop, y_target_prop = self._square_targets[idx]
                x_target = x_target_prop * self.x_range + self.x_min
                y_target = y_target_prop * self.y_range + self.y_min
                return np.array([x_target, y_target])

            def _circle_sample(self, x_center, y_center, radius):
                idx = np.random.choice(np.arange(len(self._circle_targets)))
                radius_prop, theta = self._circle_targets[idx]
                x_target = x_center + radius * radius_prop * math.sin(theta)
                y_target = y_center + radius * radius_prop * math.cos(theta)
                x_target = _clip(x_target, self.x_min, self.x_max)
                y_target = _clip(y_target, self.y_min, self.y_max)
                return np.array([x_target, y_target])

            def _prepare_info(self, observation, player, action_dict):
                """
                This method is for store information of last time step,
                which is for calculating states & rewards
                """
                '''
                prepare obs, player, action dict
                '''
                self.observation = observation
                self.player = player
                self.action_dict = action_dict

                '''
                prepare player information
                '''
                self.playerKill = np.array([player_i["kill"] for player_i in player])
                self.playerDeath = np.array([player_i["death"] for player_i in player])
                self.playerCrystal = np.array([player_i["crystal"] for player_i in player])

                '''
                get player location, rotation, hp, shield
                '''
                player_location = []
                player_rotation = []
                player_hp = []
                player_shield = []

                for i in range(len(player)):
                    player_i = player[i]
                    static_i = self.playerStatic[i]
                    assert player_i["playerId"] == static_i["playerId"]

                    x = player_i["x"]
                    y = player_i["y"]
                    player_location.append(x)
                    player_location.append(y)

                    rotation = player_i["rotation"]
                    rotation = rotation / 180.0 * np.pi
                    player_rotation.append(rotation)

                    hp = player_i["hp"]
                    shield = player_i["shield"]
                    player_hp.append(hp)
                    player_shield.append(shield)

                self.player_location = self._get_locations(player_location)
                self.player_rotation = player_rotation
                self.player_hp = player_hp
                self.player_shield = player_shield

                '''
                prepare items locations
                '''
                self.mapBox = self._get_locations(
                    observation.get("mapBox", None))
                self.mapBuff = self._get_locations(
                    observation.get("mapBuff", None))
                self.mapCrystal = self._get_locations(
                    observation.get("mapCrystal", None))
                # self.mapBullets = self._get_locations(
                #     observation.get("mapBullets", None))
                # self.mapWeapon = self._get_locations(
                #     observation.get("mapWeapon", None))

            @staticmethod
            def _log_uniform(ranges, modes, return_prop=False):
                prop = np.random.random((len(modes)))
                rewards = list()
                for prop_i, mode in zip(prop, modes):
                    _p = prop_i * (np.log(ranges[mode][1]) - np.log(ranges[mode][0])) + np.log(ranges[mode][0])
                    rewards.append(np.exp(_p))
                rewards = np.array(rewards)
                if return_prop:
                    return rewards, prop
                return rewards

            @staticmethod
            def _log_uniform_proportion(ranges, x, mode):
                a, b = ranges[mode]
                p = (np.log(x) - np.log(a)) / (np.log(b) - np.log(a))
                return _power_embed(p, 0.0, 1.0, 3, 1.0)

            @staticmethod
            def _get_locations(flatten_locations):
                if flatten_locations is not None:
                    return np.reshape(flatten_locations, (-1, 2))
                return []

            def _register_location(self, image, channel, locations, old_locations):
                for x, y in old_locations:
                    image[math.floor(x - self.x_min), math.floor(y - self.y_min), channel] = 0.0
                    image[math.ceil(x - self.x_min), math.floor(y - self.y_min), channel] = 0.0
                    image[math.floor(x - self.x_min), math.ceil(y - self.y_min), channel] = 0.0
                    image[math.ceil(x - self.x_min), math.ceil(y - self.y_min), channel] = 0.0
                for x, y in locations:
                    image[math.floor(x - self.x_min), math.floor(y - self.y_min), channel] += 1.0
                    image[math.ceil(x - self.x_min), math.floor(y - self.y_min), channel] += 1.0
                    image[math.floor(x - self.x_min), math.ceil(y - self.y_min), channel] += 1.0
                    image[math.ceil(x - self.x_min), math.ceil(y - self.y_min), channel] += 1.0
                return image

        return XGameEnv

    return decorator
