import sys
import numpy as np
import pylab
import math


t = 1
sample_size = 160
training_size = 100
center = (330, 177)
u = .005 # acceleration 
Un = np.matrix([[1/2*u*pow(t, 2)],  # x position
               [1/2*u*pow(t, 2)],   # y position
               [u*t],         # Vx
               [u*t]])        # Vy             control vector
move_noise = .1 # bug moving noise
Q = np.matrix([[pow(t, 4)/4, 0, pow(t, 3)/2, 0], 
                [0, pow(t, 4)/4, 0, pow(t, 3)/2],
                [pow(t, 3)/2, 0, pow(t, 2), 0],
                [0, pow(t, 3)/2, 0, pow(t, 2)]])*pow(move_noise, 2); # move covariance matrix

noise_x = 1;  # x direction measurement noise 
noise_y = 1;  # y direction measurement noise 
R = np.matrix([[noise_x, 0, 0, 0],
               [0, noise_y, 0, 0],
               [0, 0, noise_x, 0],
               [0, 0, 0, noise_y]])   # measurement covariance only apply on positions



# update equation
A = np.matrix([[1, 0, t, 0],
               [0, 1, 0, t],
               [0, 0, 1, 0],
               [0, 0, 0, 1]]) #state transition
B = np.matrix([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])  # control matrix
H = np.matrix([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])  # observation matrix

P = np.eye(4)  # initial covariance

    
    
        
class EKF:
    def __init__(self, A, B, H, x, P, Q, R):
        self.A = A
        self.B = B 
        self.H = H 
        self.state_n = x
        self.covariance_n = P 
        self.error_move = Q 
        self.error_measurement = R 
        
    def predict(self, Un, Zn):    
        
        predicted_state = self.A * self.state_n + self.B * Un
        predicted_cov = self.A * self.covariance_n * np.transpose(self.A) + self.error_move
        observe_cov = self.H * predicted_cov * np.transpose(self.H) + self.error_measurement
        kalman_gain = predicted_cov * np.transpose(self.H) * np.linalg.inv(observe_cov)

        if Zn != 'NA':
            observe = Zn - self.H * predicted_state
            self.state_n = predicted_state + kalman_gain * observe
        else:
            self.state_n = predicted_state
        dimesion = self.covariance_n.shape[0]
        self.covariance_n = (np.eye(dimesion)-kalman_gain * self.H) * predicted_cov
        
    def get_state_n(self):
        return self.state_n
    

class HexBug:
    def __init__(self, wall, center, radius):
        self.wall = wall
        self.center = center
        self.radius = radius
        
    def update_wall(self, x_min, x_max, y_min, y_max):
        self.wall = (x_min, x_max, y_min, y_max)

    def set_radius(self, closest):
        distance = self.distance_between(self.center, closest)
        self.radius = min(self.radius, distance)
        
    def distance_between(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def get_projection(self, samples):
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
    x = np.matrix([[init_x],
                   [init_y],
                   [u],
                   [u]])   # initial state
    
    kf = EKF(A, B, H, x, P, Q, R)
    smallest_x = largest_x = init_x
    smallest_y = largest_y = init_y
    
    for i in xrange(sample_size):
        predicted_x.append(kf.get_state_n()[0, 0])
        predicted_y.append(kf.get_state_n()[1, 0])
        
        measured_x, measured_y = points[i][0], points[i][1]
        smallest_x = min(measured_x, smallest_x)
        largest_x = max(measured_x, largest_x)
        smallest_y = min(measured_y, smallest_y)
        largest_y = max(measured_y, largest_y)
        
        if i >= training_size:
            Zn = 'NA'
        else:
            Zn = np.matrix([[measured_x],
                            [measured_y],
                            [0],
                            [0]])
        
        kf.predict(Un, Zn)
        
    xs = [p[0] for p in points[:sample_size]]
    ys = [p[1] for p in points[:sample_size]]
    pylab.plot(xs, ys,'-', predicted_x,predicted_y,':')
    pylab.xlabel('X position')
    pylab.ylabel('Y position')
    pylab.title('Measurement of hexbug')
    pylab.legend(('measured','kalman'))
    pylab.show()   

if __name__ == "__main__":
    main()
        
        