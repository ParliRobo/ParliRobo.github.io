# coding: utf-8

import tensorflow as tf
import numpy as np
from agent.env import _warp_env

from agent.NewLSTM import NewLSTM
from agent.policy_graph import warp_Model

import os
import time
import datetime
import pytz
import socketserver
import socket
import GamebotAPI_pb2
import sys


def one_step(agent_num):
    response = GamebotAPI_pb2.OnlineResponse()
    response_restart = GamebotAPI_pb2.OnlineResponseRestartGame()
    for i in range(agent_num):
        a = response_restart.step.add()
        a.move_x = 0.0
        a.move_y = 0.0
        a.camera = 0
        a.jump = 0
        a.squat = 0
        a.shoot = 0
        a.reload = 0
        a.switch = 0
        a.cure = 0
        a.time_span = 0
    response.restart_game.CopyFrom(response_restart)
    return response


def trasform(action_dict, agent_number):
    response = GamebotAPI_pb2.OnlineResponse()
    response_step = GamebotAPI_pb2.OnlineResponseStep()
    for i in range(agent_number):
        a = response_step.player.add()
        a.move_x = action_dict[i]["moveX"]
        a.move_y = action_dict[i]["moveY"]
        a.camera = action_dict[i]["camera"]
        a.jump = action_dict[i]["jump"]
        a.squat = action_dict[i]["squat"]
        a.shoot = action_dict[i]["shoot"]
        a.reload = action_dict[i]["reload"]
        a.switch = action_dict[i]["switch"]
        a.cure = action_dict[i]["cure"]
        a.time_span = action_dict[i]["timeSpan"]
    response.step.CopyFrom(response_step)
    return response


class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        # 对阿里云的slb探测进行特殊处理
        try:
            data = self.request.recv(8 * 1024 * 1024)
        except:
            return
        # 捕捉AWS行为 return
        if not data:
            return

        t = datetime.datetime.fromtimestamp(
            int(time.time()),
            pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
        print(
            "################## connected current process id:{} at time {} ##################"
            .format(os.getpid(), t))
        received_len = len(data)
        len_buf = b''
        request_buf = b''
        if received_len < 4:
            len_buf += data[:received_len]
        else:
            len_buf += data[0:4]
            request_buf += data[4:]

        disconnect = False
        begin = datetime.datetime.now()
        while received_len < 4:
            first_buf = self.request.recv(1024)
            if received_len + len(first_buf) < 4:
                len_buf += first_buf
            else:
                len_buf += first_buf[:4 - received_len]
                request_buf += first_buf[4 - received_len:]
            received_len += len(first_buf)
            end = datetime.datetime.now()
            if (end - begin).total_seconds() > 15:
                print("client disconnected!")
                disconnect = True
                break
        if disconnect == True:
            return

        assert len(len_buf) == 4, 'len_buf should be 4'
        file_size = int.from_bytes(len_buf, byteorder='little', signed=True)
        if file_size > len(request_buf):
            try:
                tmp_buf = self.request.recv(file_size - len(request_buf),
                                        socket.MSG_WAITALL)
                request_buf += tmp_buf
            except:
                print("An exception occurred when receiving data")
                return

        req = GamebotAPI_pb2.OnlineRequest()
        req.ParseFromString(request_buf)

        kwargs = shared_kwargs.copy()
        kwargs["init_observation"] = req
        kwargs["num"] = len(req.restart_game.player)

        # 返回一个空的动作
        response = one_step(len(req.restart_game.player))
        response_buf = b''
        response_data = response.SerializeToString()
        response_buf += len(response_data).to_bytes(4, byteorder='little')
        response_buf += response_data
        try:
            self.request.sendall(response_buf)
        except:
            print("An exception occurred when sending data")
            return

        # 初始化环境
        try:
            envs = Envs(**kwargs)
        except Exception as e:
            print("Exception in init envs")
            print(e)
            return

        # 恢复计算图
        sess = tf.Session()
        saver = tf.train.Saver()
        ckpt_path = np.random.choice(ckpt.all_model_checkpoint_paths)
        #print(ckpt_path)
        saver.restore(sess, ckpt_path)
        # 建立长连接 进行持续推理
        recv_buf = b''
        while True:
            # 先接收4个字节
            len_buf = b''
            begin = datetime.datetime.now()
            while len(recv_buf) < 4:
                first_buf = self.request.recv(1024)
                recv_buf += first_buf
                end = datetime.datetime.now()
                if (end - begin).total_seconds() > 15 and len(recv_buf) == 0:
                    print("client disconnected!")
                    disconnect = True
                    break
            # 若15秒仍未从客户端接收到数据，则断开与客户端的连接
            if disconnect == True:
                break
            len_buf = recv_buf[0:4]
            request_size = int.from_bytes(len_buf,
                                          byteorder='little',
                                          signed=True)
            recv_buf = recv_buf[4:]
            # 接收到一个完整的request
            begin = datetime.datetime.now()
            while len(recv_buf) < request_size:
                first_buf = self.request.recv(1024)
                recv_buf += first_buf
                end = datetime.datetime.now()
                if (end - begin).total_seconds() > 15 and len(recv_buf) == 0:
                    print("client disconnected!")
                    disconnect = True
                    break
            if disconnect == True:
                break
            request_buf = recv_buf[0:request_size]
            req = GamebotAPI_pb2.OnlineRequest()
            req.ParseFromString(request_buf)

            # 进行推理
            action_dict = envs.step(sess, model, req)

            response = trasform(action_dict, kwargs.get("num"))
            response_buf = b''
            response_data = response.SerializeToString()
            response_buf += len(response_data).to_bytes(4, byteorder='little')
            response_buf += response_data
            try:
                self.request.sendall(response_buf)
            except:
                print("An exception occurred when sending data")
                break

            # 更新recv_buf
            recv_buf = recv_buf[request_size:]
            time.sleep(0.001)


    def recv_request(self):
        # 包的格式：len+data。len为4字节int，little-endian；data为protobuf数据内容
        len_buf = b''
        request_buf = b''
        received_len = 0
        while received_len < 4:
            first_buf = self.request.recv(8 * 1024 * 1024)
            if received_len + len(first_buf) < 4:
                len_buf += first_buf
            else:
                len_buf += first_buf[:4 - received_len]
                request_buf += first_buf[4 - received_len:]
            received_len += len(first_buf)
        assert len(len_buf) == 4, 'len_buf should be 4'

        file_size = int.from_bytes(len_buf, byteorder='little', signed=True)
        tmp_buf = self.request.recv(file_size - len(request_buf),
                                    socket.MSG_WAITALL)
        request_buf += tmp_buf
        req = GamebotAPI_pb2.OnlineRequest()
        req.ParseFromString(request_buf)
        return req


class ThreadedTCPServer(socketserver.ForkingMixIn, socketserver.TCPServer):
    request_queue_size = 128
    max_children = 1024 * 8

    def collect_children(self):
        super().collect_children()

"""
graph
"""
phs = dict()
phs["image"] = tf.placeholder(
    dtype=tf.uint8, shape=[None, None, 42, 42, 17])
phs["vec"] = tf.placeholder(dtype=tf.float32, shape=[None, None, 85])
phs["meta"] = tf.placeholder(dtype=tf.float32, shape=[None, None, 51])
phs["actions"] = tf.placeholder(dtype=tf.int32, shape=[None, None, 4])
phs["rewards"] = tf.placeholder(dtype=tf.float32, shape=[None, None])
phs["state_in"] = tf.placeholder(dtype=tf.float32, shape=[None, 128 * 2])

rnn = NewLSTM(128, return_sequences=True, return_state=True, name="lstm")

Model = warp_Model()

model = Model(3, 24, 11, 0.997, rnn, False, True, False, "agent", **phs)

ckpt = tf.train.get_checkpoint_state("experiment/qmixmmo/ckpt_1/")
"""
env
"""
Envs = _warp_env()
shared_kwargs = {
    "act_space": [2, 24, 11, 2],
    "state_size": 128 * 2,
    "num": 20,
    "burn_in": 40,
    "seqlen": 40,
    "speed": 2
}

HOST, PORT = "", 8001
server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
server.serve_forever()
