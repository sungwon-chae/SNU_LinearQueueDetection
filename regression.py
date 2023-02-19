import numpy as np
from utils.plots import Annotator, colors


class QueueClassifier:
    def __init__(self, xybox_list, pre_line, def_line, reg_th, dis_th):
        self.xybox_list = xybox_list
        self.pre_line = pre_line
        self.def_line = def_line
        self.reg_th = reg_th
        self.dis_th = dis_th
        self.is_in_line = np.zeros(len(self.xybox_list))

    def classify_pedestrians(self): # classify pedestrians as being in the line or not based on distance to line
        last_idx = -1
        xy_list = self.xybox_list[:,0:2]
        for i, point in enumerate(xy_list):
            dist = self.distance_to_line(point, self.pre_line)
            if dist < self.reg_th:
                if last_idx < 0 or abs(xy_list[last_idx][0] - xy_list[i][0]) < self.dis_th:
                    self.is_in_line[i] = True
                    last_idx = i
                else:
                    break    
        return self.is_in_line    

    def queue_line(self): # build a new line based on the pedestrians classified as being in the line
        xy_list = self.xybox_list[:,0:2]
        x_in_line, y_in_line = [], []
        if sum(self.is_in_line) > 0:
            for i in range(len(self.is_in_line)):
                if self.is_in_line[i]:
                    x_in_line.append(xy_list[i,0])
                    y_in_line.append(xy_list[i,1])
            new_line = np.polyfit(x_in_line, y_in_line, 1)
        else:
            new_line = self.pre_line
        self.pre_line = new_line
        return new_line


    def distance_to_line(self, point, line): # compute the distance between a point and a line
        x, y = point
        m, b = line
        dist = abs(y - m*x - b) / np.sqrt(1 + m**2)
        return dist
