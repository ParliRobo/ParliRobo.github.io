#coding=utf-8
from socket import *
import GamebotAPI_pb2
import time
import datetime


HOST='127.0.0.1'
PORT=8001
BUFSIZ=1024
ADDR = (HOST,PORT)
tcpCliSock = socket(AF_INET,SOCK_STREAM)
tcpCliSock.connect(ADDR)
agent_number = 30

def one_observation(agent_num):
    request = GamebotAPI_pb2.OnlineRequest()
    request_step = GamebotAPI_pb2.OnlineRequestStep()
    game_ob = GamebotAPI_pb2.GameObservation()
    # 初始化GameObservation
    for i in range(4):
        game_ob.map_box.append(1.0)
        game_ob.map_crystal.append(2.0)
        game_ob.map_buff.append(3.0)
        game_ob.map_bullets.append(4.0)
        game_ob.map_zombie.append(10.0)
    request_step.observation.CopyFrom(game_ob)
    for i in range(agent_num):
        a = request_step.player.add()
        a.player_id = i
        a.player_role = 0
        a.alive_or_dead = i
        a.hp = i
        a.shield = 1
        a.hp_packs = 1
        a.x = 10.0 + i
        a.y = 10.0 + i
        a.rotation  = 10.0
        a.stand_or_squat = 1
        a.bullets = 10
        a.enemy_in_sight = 1
        a.enemy_in_shoot = 1
        a.zombie_in_shoot = 1
        a.zombie_in_sight = 1
        a.hurt = 1
        a.hurt_x = 1.0+i
        a.hurt_y = 1.0 + i
    request.step.CopyFrom(request_step)
    return request

def init_observation(agent_num):
    request = GamebotAPI_pb2.OnlineRequest()
    request_restart = GamebotAPI_pb2.OnlineRequestRestartGame()
    game_ob = GamebotAPI_pb2.InitGameObservation()
    game_ob.game_id = "001"
    game_ob.sea_horizon_height = 30.0
    for i in range(1002001):
        game_ob.map_height.append(30.0)
        game_ob.map_landscape.append(0)
        game_ob.map_reach.append(1)
    game_ob.camera_width = 120.0
    game_ob.camera_range = 50.0
    game_ob.x_min = -500.0
    game_ob.x_max = 500.0
    game_ob.y_min = -500.0
    game_ob.y_max = 500.0

    # 初始化GameObservation
    for i in range(4):
        game_ob.map_box.append(1.0)
        game_ob.map_crystal.append(2.0)
        game_ob.map_buff.append(3.0)
        game_ob.map_bullets.append(4.0)
        game_ob.map_zombie.append(10.0)
    request_restart.observation.CopyFrom(game_ob)
    for i in range(agent_num):
        a = request_restart.player.add()
        a.player_id = i
        a.player_name = "name" + str(i)
        a.weapon_damage = 1.0
        a.weapon_range = 10.0
        a.weapon_speed = 10.0
        a.weapon_bullets = 10
        a.full_hp = 800
        a.full_shield = 400
        a.full_hp_packs = 3
        a.player_role = 0
        a.alive_or_dead = 1
        a.hp = 500
        a.shield = 300
        a.hp_packs = 1
        a.x = 10.0 + i
        a.y = 10.0 + i
        a.rotation  = 0.0
        a.stand_or_squat = 1
        a.bullets = 10
        a.enemy_in_sight = 0
        a.enemy_in_shoot = 0
        a.zombie_in_shoot = 0
        a.zombie_in_sight = 0
        a.hurt = 0
        a.hurt_x = 0
        a.hurt_y = 0
    request.restart_game.CopyFrom(request_restart)
    return request



if __name__ == '__main__':
    request = init_observation(agent_number)
    request_buf = b''
    request_data = request.SerializeToString()
    request_buf += len(request_data).to_bytes(4, byteorder='little')
    request_buf += request_data
    tcpCliSock.send(request_buf)

    # 获取推理结果
    len_buf = b''
    response_buf = b''
    received_len = 0
    while received_len < 4:
        first_buf = tcpCliSock.recv(BUFSIZ)
        if received_len + len(first_buf) < 4:
            len_buf += first_buf
        else:
            len_buf += first_buf[:4 - received_len]
            response_buf += first_buf[4 - received_len:]
        received_len += len(first_buf)
    assert len(len_buf) == 4, 'len_buf should be 4'
    file_size = int.from_bytes(len_buf, byteorder='little', signed=True)
    tmp_buf = tcpCliSock.recv(file_size - len(response_buf))
    response_buf += tmp_buf

    count = 0
    while True:
        # 封装成一个推理请求
        begin = datetime.datetime.now()
        request = one_observation(agent_number)
        request_buf = b''
        request_data = request.SerializeToString()
        request_buf += len(request_data).to_bytes(4, byteorder='little')
        request_buf += request_data
        tcpCliSock.send(request_buf)
        # 获取推理结果
        len_buf = b''
        response_buf = b''
        received_len = 0
        while received_len < 4:
            first_buf = tcpCliSock.recv(BUFSIZ)
            if received_len + len(first_buf) < 4:
                len_buf += first_buf
            else:
                len_buf += first_buf[:4 - received_len]
                response_buf += first_buf[4 - received_len:]
            received_len += len(first_buf)
        assert len(len_buf) == 4, 'len_buf should be 4'
        file_size = int.from_bytes(len_buf, byteorder='little', signed=True)
        tmp_buf = tcpCliSock.recv(file_size - len(response_buf))
        response_buf += tmp_buf
        re = GamebotAPI_pb2.OnlineResponse()
        re.ParseFromString(response_buf)
        # print(re)
        end = datetime.datetime.now()
        print("The interval time is {}".format(end - begin))
        # 打印一些信息
        count +=1
        #print("The {} request".format(count))
        # 休眠0.2秒
        if count == 150:
            break
        time.sleep(0.2)
    tcpCliSock.close()
