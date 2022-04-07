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
    dist = max(0.0, min(50.0, dist))
    if dist < 5:
        dist_rate = 0.9 + dist * 0.02
    else:
        dist_rate = 1.0 - (dist - 5) * 0.6 / 45
    angle = max(0.0, min(np.pi / 6, angle))
    angle_rate = 1.0 - 0.5 * (angle / (np.pi / 6)) ** 0.33
    return dist_rate * angle_rate


def preprocessor(shape=(42, 42), landscape_n_values=6):
    def decorator(cls):
        class EightGameEnv(cls):
            def __init__(self, *args, **kwargs):
                self.trace_memory_range = {0: [0.01, 0.1]}
                self.trace_length_range = {0: [5.0, 20.0]}
                self.hit_rate_range = {0: [0.7, 0.9]}

                super(EightGameEnv, self).__init__(*args, **kwargs)
                self.rBuffRange = {0: [0.1, 1.0],
                                   1: [0.1, 1.0]}
                self.rBoxRange = {0: [0.1, 1.0],
                                  1: [0.1, 1.0]}
                # self.rBulletRange = {0: [0.02, 0.2],
                #                      1: [0.02, 0.2]}
                # self.rWeaponRange = {0: [1.0, 2.0],
                #                      1: [1.0, 2.0]}
                self.rGetCrystalRange = {0: [0.75, 2.25],
                                         1: [0.75, 2.25]}
                self.rLoseCrystalRange = {0: [0.01, 0.1],
                                          1: [0.01, 0.1]}
                self.rKillRange = {0: [75.0, 100.0],
                                   1: [1.0, 10.0]}
                self.rDeathRange = {0: [1.0, 10.0],
                                    1: [1.0, 10.0]}
                # self.rHpRange = {0: [0.0002, 0.02],
                #                  1: [0.0002, 0.02]}
                # self.rShieldRange = {0: [0.0001, 0.01],
                #                      1: [0.0001, 0.01]}
                self.rHurtRange = {0: [0.1, 1.0],
                                   1: [0.01, 0.1]}
                self.rBeHurtRange = {0: [0.01, 0.1],
                                     1: [0.01, 0.1]}
                self.rInterruptMoveRange = {0: [0.01, 0.1],
                                            1: [0.01, 0.1]}
                self.rFaceMoveRange = {0: [0.002, 0.02],
                                       1: [0.002, 0.02]}
                self.rContinuousMoveRange = {0: [0.001, 0.01],
                                             1: [0.001, 0.01]}
                self.rFinishMoveRange = {0: [0.0001, 0.001],
                                         1: [0.0001, 0.001]}
                self.rShootInShootRange = {0: [0.025, 0.25],
                                           1: [0.0001, 0.001]}
                self.rShootInSightRange = {0: [0.005, 0.05],
                                           1: [0.0001, 0.001]}
                # self.rShootNotInShootRange = {0: [0.005, 0.05],
                #                               1: [0.0001, 0.001]}
                self.rShootNotInSightRange = {0: [0.025, 0.25],
                                              1: [0.0001, 0.001]}
                self.rCameraRange = {0: [0.001, 0.01],
                                     1: [0.0001, 0.001]}

            def reset(self, init_observation):
                reset_info = super().reset(init_observation)

                restartGame = reset_info["restartGame"]

                observation = restartGame["observation"]
                player = restartGame["player"]
                policy_modes = [0] * len(player)
                self._get_rewards(policy_modes)

                # self.observation = observation
                # self.player = player

                self._process_reset_observation(observation)
                self._process_reset_player(player, policy_modes)

                # self._process_step_observation(observation)
                # s = self._process_step_player(player)

                self._prepare_info(observation, player, None)
                a = np.array([[0] * 4] * self.agent_num)
                self.action_dict = self._get_action_dict(a)
                x_target_prop = np.random.uniform(0.3, 0.7, (self.agent_num, 1))
                y_target_prop = np.random.uniform(0.3, 0.7, (self.agent_num, 1))
                x_target = x_target_prop * self.x_range + self.x_min
                y_target = y_target_prop * self.y_range + self.y_min
                self.player_target = np.concatenate([x_target, y_target], axis=-1)


                for i in range(self.agent_num):
                    self.action_dict[i]["moveX"] = self.player_location[i][0]
                    self.action_dict[i]["moveY"] = self.player_location[i][1]
                    self.action_dict[i]["moveD"] = self.x_range + self.y_range

            def inference_request_process(self, inference_request):
                s, r, done, info = super().inference_step(inference_request)
                return s, r, done, info

            def step(self, s, r, done, info, action_dict):
                step = s["step"]
                observation = step["observation"]
                player = step["player"]

                rewards = self._process_step_reward(observation, player, action_dict)
                alphas = np.ones((len(player), 3), np.float32) / 3.0
                r = np.array(
                    [(alphas_i[0] * rewards_i["move"] + alphas_i[1] * rewards_i["pick"]
                      + alphas_i[2] * rewards_i["fight"] + rewards_i["general"])
                     for rewards_i, alphas_i in zip(rewards, alphas)], np.float32)

                self._process_step_observation(observation)
                states = self._process_step_player(player, action_dict)
                s = []
                for s_i, r_i in zip(states, rewards):
                    ss = s_i
                    ss.update(r_i)
                    s.append(ss)

                self._prepare_info(observation, player, action_dict)

                done = done["all"]

                return s, r, done, info

            def _set_hit_rate(self):
                self.hit_rate = self._log_uniform(
                    self.hit_rate_range, [0] * self.agent_num)

            def _get_rewards(self, policy_modes):
                self.trace_memory = self._log_uniform(self.trace_memory_range, [0] * self.agent_num)
                self.trace_length = self._log_uniform(self.trace_length_range, [0] * self.agent_num)
                self.trace_rate = np.exp(np.log(self.trace_memory) / self.trace_length)

                self.rBuff = self._log_uniform(self.rBuffRange, policy_modes)
                self.rBox = self._log_uniform(self.rBoxRange, policy_modes)
                # self.rBullet = self._log_uniform(self.rBulletRange, policy_modes)
                # self.rWeapon = self._log_uniform(self.rWeaponRange, policy_modes)
                self.rGetCrystal = self._log_uniform(self.rGetCrystalRange, policy_modes)
                self.rLoseCrystal = self._log_uniform(self.rLoseCrystalRange, policy_modes)
                self.rKill = self._log_uniform(self.rKillRange, policy_modes)
                self.rDeath = self._log_uniform(self.rDeathRange, policy_modes)
                # self.rHp = self._log_uniform(self.rHpRange, policy_modes)
                # self.rShield = self._log_uniform(self.rShieldRange, policy_modes)
                self.rHurt = self._log_uniform(self.rHurtRange, policy_modes)
                self.rBeHurt = self._log_uniform(self.rBeHurtRange, policy_modes)
                self.rInterruptMove = self._log_uniform(self.rInterruptMoveRange, policy_modes)
                self.rFaceMove = self._log_uniform(self.rFaceMoveRange, policy_modes)
                self.rContinuousMove = self._log_uniform(self.rContinuousMoveRange, policy_modes)
                # self.rFinishMove = self._log_uniform(self.rFinishMoveRange, policy_modes)
                self.rShootInShoot = self._log_uniform(self.rShootInShootRange, policy_modes)
                self.rShootInSight = self._log_uniform(self.rShootInSightRange, policy_modes)
                # self.rShootNotInShoot = self._log_uniform(self.rShootNotInShootRange, policy_modes)
                self.rShootNotInSight = self._log_uniform(self.rShootNotInSightRange, policy_modes)
                self.rCamera = self._log_uniform(self.rCameraRange, policy_modes)

            def _get_action_dict(self, a):
                action_dict = dict()
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
                        d["shoot"] = 1
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
                self.polar_r = np.linspace(0, self.cameraRange * 1.1, shape[0] + 1)[1:]
                self.polar_theta = np.linspace(-self.cameraWidth / 2, self.cameraWidth / 2, shape[1])
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
                add 7 channels for loc, box, buff, #bullets, crystal, #weapon, poison
                '''
                self.mapInfo = np.concatenate(
                    [np.zeros((self.x_range, self.y_range, 5)),
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
                    self.mapInfo, 3, mapBuff, prev_mapBuff)

                prev_mapCrystal = getattr(self, "mapCrystal", [])
                self.mapInfo = self._register_location(
                    self.mapInfo, 4, mapCrystal, prev_mapCrystal)

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
                             + xs[:, None, None] - self.x_min
                             - 0.1 * self.cameraRange * np.sin(rotations[:, None, None]))
                camera_ys = (self.polar_r[None, :, None] * np.cos(camera_theta)
                             + ys[:, None, None] - self.y_min
                             - 0.1 * self.cameraRange * np.cos(rotations[:, None, None]))

                xs = np.array(np.round(camera_xs), np.int32)
                ys = np.array(np.round(camera_ys), np.int32)
                xs = np.minimum(np.maximum(xs, 0), self.x_range - 1)
                ys = np.minimum(np.maximum(ys, 0), self.y_range - 1)

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
                    info.extend(_soft_rot_embed(sin, cos, 4, 0.0))

                    target_x, target_y = self.player_target[i]
                    target_x, target_y = float(target_x), float(target_y)
                    self._register_minimap([(target_x, target_y)], 3)
                    info.extend(_soft_rot_embed(target_x - x, target_y - y, 8, rotations[i]))

                    target_distance = ((target_x - x) ** 2 + (target_y - y) ** 2) ** 0.5
                    proportion = target_distance / ((self.x_range + self.y_range) / 2)
                    proportion = _clip(proportion, 0.0, 1.0)
                    info.extend(_binary_embed(proportion, 0.0, 1.0, 4, 0.5))

                    nearest_x, nearest_y = player_location[nearest[i]]
                    nearest_x, nearest_y = float(nearest_x), float(nearest_y)
                    info.extend(_soft_rot_embed(nearest_x - x, nearest_y - y, 4, rotations[i]))
                    nearest_dist = ((nearest_x - x) ** 2 + (nearest_y - y) ** 2) ** 0.5
                    info.extend(_power_embed(_clip(
                        nearest_dist, 0.0, self.cameraRange), 0.0, self.cameraRange, 3, 1.0))

                    moveX, moveY = action_dict[i]["moveX"], action_dict[i]["moveY"]
                    info.extend(_soft_rot_embed(moveX - x, moveY - y, 4, rotations[i]))

                    distance = ((moveX - x) ** 2 + (moveY - y) ** 2) ** 0.5
                    moveD = action_dict[i]["moveD"]
                    unfinish = min(distance / max(moveD, 1e-8), 1.0)
                    info.extend(_power_embed(unfinish, 0.0, 1.0, 3, 1.0))

                    hurt = player_i["hurt"]
                    hurtX = player_i["hurtX"]
                    hurtY = player_i["hurtY"]
                    info.append(hurt)
                    if hurt:
                        relative_theta_info = _soft_rot_embed(hurtX - x, hurtY - y, 4, rotations[i])
                    else:
                        relative_theta_info = [0.0] * 4
                    info += relative_theta_info

                    playerStepVec = info

                    '''
                    get player image info
                    '''
                    image = self.mapInfo[[xs[i], ys[i]]]
                    playerStepImage = np.array(100 * np.concatenate(
                        [image, self.mini_map], axis=-1), dtype=np.uint8)

                    playerState.append({"image": playerStepImage,
                                        "vec": np.array(playerStepVec, np.float32),
                                        "meta": np.array(self.playerMeta[i], np.float32)})
                return playerState

            def _process_step_reward(self, observation, player, action_dict):
                n_players = len(player)
                assert n_players == self.agent_num
                rewards = {"move": np.zeros(n_players, np.float32),
                           "pick": np.zeros(n_players, np.float32),
                           "fight": np.zeros(n_players, np.float32),
                           "general": np.zeros(n_players, np.float32)}

                '''
                task:
                    0 : move
                    1 : pick
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
                rewards["move"] += tmp * 0.428571
                rewards["pick"] += tmp * 0.428571

                rewards["general"] -= self.rDeath * (playerDeath - self.playerDeath)

                tmp = self.rGetCrystal * np.maximum(
                    playerCrystal - self.playerCrystal, np.zeros(n_players))
                rewards["fight"] += tmp * 0.428571
                rewards["move"] += tmp * 0.428571
                rewards["pick"] += tmp * 2.142857

                rewards["general"] -= self.rLoseCrystal * np.maximum(
                    self.playerCrystal - playerCrystal, np.zeros(n_players))

                # playerHp = np.array([player_i["hp"] for player_i in player])
                # playerShield = np.array([player_i["shield"] for player_i in player])
                hurt = np.array([player_i["hurt"] for player_i in player])
                hurtId = np.array([player_i["hurtId"] for player_i in player])
                for i, hurt_i in enumerate(hurt):
                    if hurt_i:
                        # hp_loss = max(self.player_hp[i] - playerHp[i], 0)
                        # shield_loss = min(self.player_shield[i] - playerShield[i], 0)
                        # r[i] -= self.rHp[i] * hp_loss
                        # r[i] -= self.rShield[i] * shield_loss
                        rewards["general"][i] -= self.rBeHurt[i]
                        # r[self.id2idx[hurtId[i]]] += self.rHp[self.id2idx[hurtId[i]]] * hp_loss
                        # r[self.id2idx[hurtId[i]]] += self.rShield[self.id2idx[hurtId[i]]] * shield_loss
                        tmp = self.rHurt[self.id2idx[hurtId[i]]]
                        rewards["fight"][self.id2idx[hurtId[i]]] += tmp * 2.142857
                        rewards["move"][self.id2idx[hurtId[i]]] += tmp * 0.428571
                        rewards["pick"][self.id2idx[hurtId[i]]] += tmp * 0.428571

                '''
                get new items locations & update r
                '''
                mapBox = self._get_locations(
                    observation.get("mapBox", None))
                mapBuff = self._get_locations(
                    observation.get("mapBuff", None))
                # mapBullets = self._get_locations(
                #     observation.get("mapBullets", None))
                # mapWeapon = self._get_locations(
                #     observation.get("mapWeapon", None))

                pick_box = []
                for box in self.mapBox:
                    if box not in mapBox:
                        pick_box.append(box)
                pick_box = np.reshape(pick_box, (-1, 2))
                distance = np.sum(np.square(
                    pick_box[:, None, :] * self.player_location[None, :, :]),
                    axis=-1)
                pick = np.sum(distance < 16, axis=0)
                tmp = self.rBox * pick
                rewards["fight"] += tmp * 0.428571
                rewards["move"] += tmp * 0.428571
                rewards["pick"] += tmp * 2.142857

                pick_buff = []
                for buff in self.mapBuff:
                    if buff not in mapBuff:
                        pick_buff.append(buff)
                pick_buff = np.reshape(pick_buff, (-1, 2))
                distance = np.sum(np.square(
                    pick_buff[:, None, :] * self.player_location[None, :, :]),
                    axis=-1)
                pick = np.sum(distance < 16, axis=0)
                tmp = self.rBuff * pick
                rewards["fight"] += tmp * 0.428571
                rewards["move"] += tmp * 0.428571
                rewards["pick"] += tmp * 2.142857

                '''
                get moveX moveY & update r
                '''
                for i in range(n_players):
                    x, y = self.player_location[i]
                    rotation = player[i]["rotation"] / 180.0 * np.pi
                    if (action_dict[i]["moveX"] != self.action_dict[i]["moveX"]) and (
                            action_dict[i]["moveY"] != self.action_dict[i]["moveY"]):
                        moveX = self.action_dict[i]["moveX"]
                        moveY = self.action_dict[i]["moveY"]
                        moveD = self.action_dict[i]["moveD"]
                        # if np.sqrt(np.square(moveX - x) + np.square(moveY - y)) < 0.3 * moveD:
                        #     r[i] += self.rFinishMove[i]
                        if ((moveX - x) ** 2 + (moveY - y) ** 2) ** 0.5 > 0.6 * moveD:
                            rewards["general"][i] -= self.rInterruptMove[i]

                    move_distance = ((player[i]["x"] - x) ** 2 + (player[i]["y"] - y) ** 2) ** 0.5
                    if move_distance < 0.1:
                        rewards["general"][i] -= 0.5

                    delta_x = (player[i]["x"] - x) / 10.0  # * min(1.0 / max(move_distance, 1e-8), 1.0)
                    delta_y = (player[i]["y"] - y) / 10.0  # * min(1.0 / max(move_distance, 1e-8), 1.0)
                    '''
                    cts move & face move
                    '''
                    cosine = (delta_x * self.playerXtrace[i] + delta_y * self.playerYtrace[i])
                    rewards["fight"] += self.rContinuousMove[i] * (99.0 + cosine) / 100.0 * 0.428571
                    rewards["move"] += self.rContinuousMove[i] * (0.25 + cosine) / 1.25 * 2.142857
                    rewards["pick"] += self.rContinuousMove[i] * (2.5 + cosine) / 3.5 * 0.428571

                    self.playerXtrace[i] = delta_x + self.trace_rate[i] * self.playerXtrace[i]
                    self.playerYtrace[i] = delta_y + self.trace_rate[i] * self.playerYtrace[i]
                    cosine = (np.sin(rotation) * self.playerXtrace[i]
                              + np.cos(rotation) * self.playerYtrace[i])
                    rewards["fight"] += self.rFaceMove[i] * (1 - np.abs(cosine)) * 0.428571
                    rewards["move"] += self.rFaceMove[i] * (0.0 + cosine) / 1.0 * 2.142857
                    rewards["pick"] += self.rFaceMove[i] * (1.5 + cosine) / 2.5 * 0.428571

                    '''
                    target move
                    '''
                    has_poison = observation.get("hasPoison", False)
                    poison_radius = observation.get("poisonRadius", 999999.0)
                    poison_center_x = observation.get("poisonCenterX", 0.0)
                    poison_center_y = observation.get("poisonCenterY", 0.0)
                    old_locations = np.array(self.player_location[i])
                    new_locations = np.array([player[i]["x"], player[i]["y"]])
                    old_distance = np.sum((self.player_target[i] - old_locations) ** 2) ** 0.5
                    new_distance = np.sum((self.player_target[i] - new_locations) ** 2) ** 0.5
                    distance_reduce = old_distance - new_distance
                    if has_poison:
                        tmp = distance_reduce * (self.rContinuousMove[i] + self.rFaceMove[i]) * 8.0
                    else:
                        tmp = distance_reduce * (self.rContinuousMove[i] + self.rFaceMove[i])
                    rewards["fight"] += tmp * 0.428571
                    rewards["move"] += tmp * 2.142857
                    rewards["pick"] += tmp * 0.428571

                    not_safe_target = False
                    if has_poison:
                        dist = np.sum((self.player_target[i] - np.array([
                            poison_center_x, poison_center_y])) ** 2) ** 0.5
                        not_safe_target = np.array(dist > poison_radius)
                    if new_distance < 25 or not_safe_target:
                        if not_safe_target:
                            p = np.linspace(-0.5, 0.5, 20) ** 2
                            p /= np.sum(p)
                            choices = np.linspace(0.0, 1.0, 20)
                            radius_prop = np.random.choice(choices, p=p)
                            theta = np.random.uniform(0, np.pi * 2)
                            x_target = poison_center_x + poison_radius * radius_prop * math.sin(theta)
                            y_target = poison_center_y + poison_radius * radius_prop * math.cos(theta)
                            x_target = _clip(x_target, self.x_min, self.x_max)
                            y_target = _clip(y_target, self.y_min, self.y_max)
                            self.player_target[i] = np.array([x_target, y_target])
                        elif np.random.random() < 0.6:
                            p = np.linspace(-0.5, 0.5, 20) ** 2
                            p /= np.sum(p)
                            choices = np.linspace(0.3, 0.7, 20)
                            x_target_prop, y_target_prop = np.random.choice(choices, 2, p=p)
                            x_target = x_target_prop * self.x_range + self.x_min
                            y_target = y_target_prop * self.y_range + self.y_min
                            self.player_target[i] = np.array([x_target, y_target])
                        else:
                            j = np.random.randint(0, self.agent_num)
                            self.player_target[i] = np.array(self.player_location[j])

                '''
                whether shoot or not & update r
                '''
                for i in range(n_players):
                    if action_dict[i]["shoot"] == 1:
                        inShoot = self.player[i]["enemyInShoot"] or self.player[i]["zombieInShoot"]
                        inSight = self.player[i]["enemyInSight"] or self.player[i]["zombieInSight"]
                        if inShoot:
                            rewards["general"][i] += self.rShootInShoot[i]
                        # else:
                        #     r[i] -= self.rShootNotInShoot[i]
                        elif inSight:
                            rewards["general"][i] += self.rShootInSight[i]
                        else:
                            rewards["general"][i] -= self.rShootNotInSight[i]

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
                        if player[i]["hurt"]:
                            rewards["fight"][i] += self.rCamera[i]
                        rewards["move"][i] -= self.rCamera[i] * np.sin(d * np.pi / 16)
                        rewards["pick"][i] -= self.rCamera[i] * np.sin(d * np.pi / 16)

                '''
                keep distance from nearest player
                '''
                # player_location = [[player_i["x"], player_i["y"]] for player_i in player]
                # player_location = np.array(player_location)
                # player_distance = np.sum((player_location[:, None, :] -
                #                           player_location[None, :, :]) ** 2,
                #                          axis=-1) ** 0.5
                # player_distance += 65535 * np.eye(self.agent_num)
                # nearest_distance = np.min(player_distance, axis=-1)
                #
                # rewards["general"] -= np.maximum(1 - nearest_distance / 5.0, 0) ** 2 * 2

                rewards = [{k: v[i] for k, v in rewards.items()} for i in range(n_players)]
                return rewards

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
            def _log_uniform(ranges, modes):
                return np.array([np.exp(np.random.uniform(np.log(
                    ranges[mode][0]), np.log(ranges[mode][1]))) for mode in modes])

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

        return EightGameEnv

    return decorator
