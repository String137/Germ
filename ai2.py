import sys
import random

import tensorflow as tf
import keras
import numpy as np
import random
import pickle
import sys
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from collections import deque
from keras.layers import Reshape
from keras import backend as K
from keras.callbacks import TensorBoard
from time import time

BOARD_ROWS = 7
BOARD_COLS = 7

class seven:
    def aTp(self,Action):
        position = np.zeros(4)
        Action = int(Action)
        for i in range(4):
            position[3-i] = Action % BOARD_COLS
            Action = Action // BOARD_COLS
            position = [int(i) for i in position]
        return position
        
class Player:
    def __init__(self,isHuman,playerSymbol):
        self.isHuman=isHuman
        self.model = self.build_model()
        self.playerSymbol = playerSymbol
    
    def build_model(self): # DQN 모델을 생성한다.
        model = Sequential()
        model.add(Conv2D(16, (3, 3), padding = 'valid', input_shape=(7, 7, 1), dtype='float32'))
        model.add(LeakyReLU(0.3))
        model.add(Conv2D(16, (3, 3), padding = 'valid', input_shape=(5, 5, 1), dtype='float32'))
        model.add(LeakyReLU(0.3))
        model.add(Flatten())
        model.add(Dense(64 * BOARD_COLS * BOARD_COLS, dtype='float32'))
        model.add(LeakyReLU(0.3))
        model.add(Dense(BOARD_COLS**4, dtype='float32'))
        model.add(LeakyReLU(0.3))
        model.add(Reshape((BOARD_ROWS**4,), dtype='float32'))
        return model

    def availableActions(self,state): # 가능한 행동들을 반환한다.
        Actions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if state[0][i][j][0] == self.playerSymbol:
                    for ii in range(-2,3):
                        for jj in range(-2,3):
                            if ii == 0 and jj == 0:
                                continue
                            if i + ii < 0 or i + ii >= BOARD_ROWS or j + jj < 0 or j + jj >= BOARD_COLS:
                                continue
                            if state[0][i + ii][j + jj][0] == 0:
                                act = i
                                act = act*BOARD_COLS + j
                                act = act*BOARD_COLS + i + ii
                                act = act*BOARD_COLS + j + jj
                                Actions.append(act)
        return Actions
    
    def isAvailableAction(self, state, Action): # 가능한 행동인지?
        position = np.zeros(4)
        Action = int(Action)
        for i in range(4):
            position[3-i] = Action % BOARD_COLS
            Action = Action // BOARD_COLS
            position = [int(i) for i in position]
        return state[position[0]][position[1]]==self.playerSymbol and state[position[2]][position[3]]==0

    
    def getAction(self,state):
        if self.isHuman:
            print("original row, col, target row, col")
            ro = int(input())
            co = int(input())
            rt = int(input())
            ct = int(input())
            if not ro>=0 and ro<7 and co>=0 and co<7 and rt>=0 and rt<7 and ct>=0 and ct<7:
                return None
            act = ro*7*7*7+co*7*7+rt*7+ct
            if not self.isAvailableAction(state,act):
                return None
            return act
        else:
            q_val = self.model.predict(state.reshape(1,BOARD_ROWS, BOARD_COLS, 1))
            avac = self.availableActions(state)
#             print(avac)
            if not avac:
                return None
            avq = np.zeros(len(avac))
            for i,a in enumerate(avac):
                avq[i]=a
                p=seven.aTp(seven,a)
#                 print("(",p[0],",",p[1],") --> (",p[2],",",p[3],")   :  ",q_val[0,a])
#             for i in range(7):
#                 for j in range(7):
#                     for k in range(7):
#                         for l in range(7):
#                             print("(",i,",",j,") --> (",k,",",l,")   :  ","{0:6f}".format(q_val[0,343*i+49*j+7*k+l]))
            return avq[np.argmax(q_val[0,avac])]

    def load(self, model_filepath):
        self.model = keras.models.load_model(model_filepath)
        
if __name__ == "__main__":

    input_str = sys.stdin.read()

#     file = open("file.txt")
#     input_str = file.read()
#     print(input_str)

    # 입력 예시
    # READY 1234567890.1234567 (입력시간)
    # "OK" 를 출력하세요.
    
    if input_str.startswith("READY"):
        # 출력
        sys.stdout.write("OK")

    # 입력 예시
    # PLAY
    # 2 0 0 0 0 0 1
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # 1 0 0 0 0 0 2
    # 1234567890.1234567 (입력시간)

    # AI의 액션을 출력하세요.
    # 출력 예시 : "0 0 2 2"
    elif input_str.startswith("PLAY"):
        player = 1 if __file__[2]=="2" else -1
#         player = -1
        board = []
        actions = {} # { key: piece(start position), value: list of position(destination position) }
#         print(player)
        
        # make board
        input_lines = input_str.split("\n")
        for i in range(7):
            board.append(list(map(int, input_lines[i+1].split(" "))))
        board = np.array(list(map(lambda row: list(map(lambda x: 2*x-3 if x!=0 else 0, row)), board)))
#         print(board)
        state = board.reshape((1,7,7,1))
#         print(state)

        nubjuk = Player(False,player)
        nubjuk.load("model/ep100k04/model")
        action = nubjuk.getAction(state)
#         print(action)
        
        if action is None:
            p = random.choice([])
            sys.stdout.write(f"{p[0]} {p[1]} {p[2]} {p[3]}")
        else:
            position = np.zeros(2)
            piece = np.zeros(2)
            position[1] = action % 7
            action = action // 7
            position[0] = action%7
            action = action // 7
            piece[1] = action % 7
            piece[0] = action // 7

            # 출력
            sys.stdout.write(f"{int(piece[0])} {int(piece[1])} {int(position[0])} {int(position[1])}")
