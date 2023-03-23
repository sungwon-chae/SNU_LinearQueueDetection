import numpy as np
from utils.plots import Annotator, colors
import cv2


class QueueClassifier:
    def __init__(self, def_line, reg_th, dis_th, start=1):
        self.reg_line = def_line
        self.def_line = def_line
        self.reg_th = reg_th
        self.dis_th = dis_th
        self.start = start % 4
        self.in_queue_list = []
        self.is_in_line = np.array([])

    def classify_pedestrians(self, xybox, im): # classify pedestrians as being in the line or not based on distance to line
        if self.start == 0:     # left
            xybox = xybox[xybox[:,0].argsort()]
        elif self.start == 1:   # top
            xybox = xybox[xybox[:,1].argsort()]
        elif self.start == 2:   # right
            xybox = xybox[(-xybox[:,0]).argsort()]
        else:                   # bottom
            xybox = xybox[(-xybox[:,1]).argsort()]
        
        SP = [410/720, 1160/1280]
        last_idx = -1
        xy_list = xybox[:,0:2]
        self.is_in_line = np.zeros(len(xy_list), dtype=bool)
        
        for i, point in enumerate(xy_list):
            dist = self.distance_to_line(point, self.reg_line)
            p1, p2 = (int(xybox[i][2]), int(xybox[i][3])), (int(xybox[i][4]), int(xybox[i][5]))
            thr_coef = 0.1 + 0.9 * xybox[i][1]
            reg_th = self.reg_th * thr_coef
            dis_th = self.dis_th * thr_coef 
            clr = (0, 0, 0)   #black
            thk = 1
            if dist < reg_th:               
                if last_idx == -1:
                    if xy_list[i][1] < SP[1] and abs(SP[0] - xy_list[i][0]) < 0.05:
                        self.is_in_line[i] = True
                        last_idx = i
                        clr = (0, 255, 0)
                        thk = 2
                elif np.linalg.norm(xy_list[last_idx] - xy_list[i]) < dis_th:
                    self.is_in_line[i] = True
                    last_idx = i
                    clr = (255, 255, 0) # mint
                    thk = 2
                else:
                    break   
            cv2.rectangle(im, p1, p2, clr, thickness=thk, lineType=cv2.LINE_AA)
    
        if len(self.in_queue_list) >= 5:
            del self.in_queue_list[0]
        self.in_queue_list.append(xybox[self.is_in_line==True])
        return self.in_queue_list[-1]   

    def queue_line(self): # build a new line based on the pedestrians classified as being in the line
        x_in_line, y_in_line = [], []
        if len(self.in_queue_list) >= 3:
            for i in range(len(self.in_queue_list)):
                x_in_line += self.in_queue_list[i][:,0].tolist()
                y_in_line += self.in_queue_list[i][:,1].tolist()
            if len(x_in_line) > 0:
                reg_line = np.polyfit(y_in_line, x_in_line, 1)
                print(reg_line)
                print(self.def_line)
                self.reg_line = (reg_line * 0.7 + self.def_line * 0.3)
        return self.reg_line


    def distance_to_line(self, point, line): # compute the distance between a point and a line
        x, y = point
        m, b = line
        dist = abs(x - m*y - b) / np.sqrt(1 + m**2)
        return dist
    
