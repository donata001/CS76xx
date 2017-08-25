import sys
import numpy as np
import pylab
import math


t = 2.0/60
sample_size = 1800
training_size = 1800
center = (330, 177)
radius = 50
u = .005 # acceleration 
turning_sensitivity = 0.01
avg_v = 2/t
avg_turn_rate = 5/t

Un = np.matrix([[1/2*u*pow(t, 2)],  # x position
               [1/2*u*pow(t, 2)],   # y position
               [0],                 # heading
               [u*t],               # velocity
               [avg_turn_rate]])              

move_noise = .1 # bug moving noise
Q = np.diag([0.05*t**2, 0.05*t**2, 0.05*t**2, avg_v**2*t**2, avg_turn_rate**2*t**2])

noise_x = 1  # x direction measurement noise 
noise_y = 1  # y direction measurement noise 
noise_tr = 1 # turn rate measurement noise
noise_heading = 1
noise_v = 1
R = np.matrix([[noise_x, 0, 0, 0],
               [0, noise_y, 0, 0],
               [0, 0, noise_v, 0],
               [0, 0, 0, noise_tr]])   # measurement covariance only apply on positions



# update equation
B = np.matrix([[1, 0, 0, 0, 0],
               [0, 1, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1]])  # control matrix
H = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 1.0]])   # observation matrix

P = np.eye(5)*10  # initial covariance

    
    
        

    
class EKF:
    def __init__(self, B, H, x, P, Q, R):
        self.B = B 
        self.H = H 
        self.state_n = x
        self.covariance_n = P 
        self.error_move = Q 
        self.error_measurement = R 
        self.JA = None
        self.Zn = None
        
    def predict(self, Zn):  
        Xn = self.get_next_Xn(self.state_n)
        print Xn
        JA = self.get_next_JA(Xn)
        self.JA = JA
        
        hx = np.matrix([[Xn[0, 0]],
                        [Xn[1, 0]],
                        [Xn[3, 0]],
                        [Xn[4, 0]]])      
        
        
        predicted_cov = self.JA * self.covariance_n * np.transpose(self.JA) + self.error_move
        observe_cov = self.H * predicted_cov * np.transpose(self.H) + self.error_measurement
        kalman_gain = (predicted_cov * np.transpose(self.H)) * np.linalg.inv(observe_cov)

        if Zn != 'NA':
            Zn = Zn.reshape(self.H.shape[0],1)
            observe = Zn - hx
            self.state_n = Xn + kalman_gain * observe
        else:
            self.state_n = Xn
            
        dimesion = self.covariance_n.shape[0]
        self.covariance_n = (np.eye(dimesion)-kalman_gain * self.H) * predicted_cov

        
    def get_state_n(self):
        return self.state_n

    
    def get_next_JA(self, Xn):
                    
        a13 = (Xn[3]/Xn[4]) * (np.cos(Xn[4]*t+Xn[2]) - np.cos(Xn[2]))
        a14 = (1.0/Xn[4]) * (np.sin(Xn[4]*t+Xn[2]) - np.sin(Xn[2]))
        a15 = (t*Xn[3]/Xn[4])*np.cos(Xn[4]*t+Xn[2]) - (Xn[3]/Xn[4]**2)*(np.sin(Xn[4]*t+Xn[2]) - np.sin(Xn[2]))
        a23 = (Xn[3]/Xn[4]) * (np.sin(Xn[4]*t+Xn[2]) - np.sin(Xn[2]))
        a24 = (1.0/Xn[4]) * (-np.cos(Xn[4]*t+Xn[2]) + np.cos(Xn[2]))
        a25 = (t*Xn[3]/Xn[4])*np.sin(Xn[4]*t+Xn[2]) - (Xn[3]/Xn[4]**2)*(-np.cos(Xn[4]*t+Xn[2]) + np.cos(Xn[2]))
        JA = np.matrix([[1.0, 0.0, a13, a14, a15],
                        [0.0, 1.0, a23, a24, a25],
                        [0.0, 0.0, 1.0, 0.0, t],
                        [0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0]])
        return JA
    
    def get_next_Xn(self, Xn):
        
        if abs(Xn[4, 0]) < turning_sensitivity:
            Xn[0] = Xn[0] + Xn[3]*t * np.cos(Xn[2])
            Xn[1] = Xn[1] + Xn[3]*t * np.sin(Xn[2])
            Xn[4] = 0.0000001 
        else:
            Xn[0] = Xn[0] + (Xn[3]/Xn[4]) * (np.sin(Xn[4]*t+Xn[2]) - np.sin(Xn[2]))
            Xn[1] = Xn[1] + (Xn[3]/Xn[4]) * (-np.cos(Xn[4]*t+Xn[2])+ np.cos(Xn[2]))
            Xn[2] = (Xn[2] + Xn[4]*t + np.pi) % (2.0*np.pi) - np.pi  
    
        return Xn
        
class HexBug:
    def __init__(self, wall, center, radius):
        self.wall = wall
        self.center = center
        self.radius = radius
        
    def update_wall(self, x_min, x_max, y_min, y_max):
        self.wall = (x_min, x_max, y_min, y_max)

    def update_radius(self, closest):
        distance = self.distance_between(self.center, closest)
        self.radius = min(self.radius, distance)
        
    def distance_between(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def get_velocity(self, samples):
        # to do, wall bounce and corner stuck
        s = self.distance_between(samples[0], samples[1])
        return s*1.0/t
    
    def get_heading(self, samples):
        pt1, pt2 = samples
        try:
            heading = math.atan2(pt2[1]-pt1[1],pt2[0]-pt1[0])
        except Exception :
            heading = math.pi/2 if pt2[1]>pt1[1] else - math.pi/2
        return heading
        
    def get_turn_rate(self, samples):
        # this needs at least 3 points
        pt0, pt1, pt2 = samples
        pylab.show()   
        v01_x = pt1[0] - pt0[0]
        v01_y = pt1[1] - pt0[1]
        v12_x = pt2[0] - pt1[0]
        v12_y = pt2[1] - pt1[1]
        d01 = self.distance_between(pt0, pt1)
        d12 = self.distance_between(pt1, pt2)
        try:
            turning_angle = math.acos((v01_x*v12_x+v01_y*v12_y)/d01/d12)
        except:
            turning_angle = 0
            return 0
        else:        
            # check clockwise or counterclockwise turn        
            clockwise = math.atan2(v01_y, v01_x) > math.atan2(v12_y, v12_x)
            turning_angle = -1*turning_angle if clockwise else turning_angle
            return turning_angle/2.0/t
    
    def is_hit_wall(self, position):
        return
        

def get_training_data():
    pts = []
    with open('training_data.txt', 'r') as f:
        for line in f.readlines()[:sample_size]:
            pt = line.strip('\n').split(',')
            pts.append((int(pt[0]), int(pt[1])))
    return pts
    
    
def main():
    predicted_x = []
    predicted_y = []
    
    points = get_training_data()
    init_x, init_y = points[0][0], points[0][1]
    x = np.matrix([[init_x],     # x position
                   [init_y],     # y position
                   [0],          # heading
                   [0.1],         # velocity
                   [0]])         # turn rate
    
    kf = EKF(B, H, x, P, Q, R)
    x_min = x_max = init_x
    y_min = y_max = init_y
    bug = HexBug((x_min, x_max, y_min, y_max), center, radius)
    
    for i in xrange(2, sample_size):
        
        predicted_x.append(kf.get_state_n()[0, 0])
        predicted_y.append(kf.get_state_n()[1, 0])
    
        measured_x, measured_y = points[i][0], points[i][1]
        x_min = min(measured_x, x_min)
        x_max = max(measured_x, x_max)
        y_min = min(measured_y, y_min)
        y_max = max(measured_y, y_max)
        
        bug.update_wall(x_min, x_max, y_min, y_max)
        bug.update_radius(points[0])
        print i
        if i >= training_size:
            Zn = 'NA'
        else:
            v = bug.get_velocity([points[i-1], points[i]])
            #heading = bug.get_heading([points[i-1], points[i]])
            turn_rate = bug.get_turn_rate([points[i-2], points[i-1], points[i]])
            Zn = np.matrix([[measured_x],
                            [measured_y],
                            [v],
                            [turn_rate]])

        
        kf.predict(Zn)
        
    xs = [p[0] for p in points[:sample_size]]
    ys = [p[1] for p in points[:sample_size]]
    #print predicted_x[-100:]
    #pylab.plot(xs[-80:], ys[-80:],'-', predicted_x[-80:],predicted_y[-80:],':')
    pylab.plot(xs, ys,'-', predicted_x,predicted_y,':')
    pylab.xlabel('X position')
    pylab.ylabel('Y position')
    pylab.title('Measurement of hexbug')
    pylab.legend(('measured','kalman'))
    pylab.show()   



if __name__ == "__main__":
    main()
        
        