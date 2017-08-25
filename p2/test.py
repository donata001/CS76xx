import sys
import numpy as np
import math
import random
import pylab


t = 1
sample_size = 1800
training_size = 1740
center = (330, 177)
radius = 50
u = .005 # acceleration 
tr = .01 # turn rate
capped_tr = 0.08

Un = np.matrix([[1/2*u*pow(t, 2)],  # x position
               [1/2*u*pow(t, 2)],   # y position
               [0],         # heading
               [u*t],       # v
               [0]])        # turn rate             control vector

move_noise = .1 # bug moving noise
Q = np.matrix([[pow(t, 4)/4, 0, pow(t, 3)/2, 0.5*pow(t, 3), 0.5*pow(t, 2)], 
                [0, pow(t, 4)/4, 0.5*pow(t, 3), pow(t, 3)/2, 0.5*pow(t, 2)],
                [pow(t, 3)/2, pow(t,3)/2, pow(t, 2), pow(t,2), t],
                [pow(t, 3)/2, pow(t, 3)/2, t**2, pow(t, 2), t],
                [pow(t,2)/2, pow(t,2)/2, t, t, 1]])*pow(move_noise, 2); # move covariance matrix

noise_x = 1;  # x direction measurement noise 
noise_y = 1;  # y direction measurement noise 
R = np.matrix([[noise_x, 0],
               [0, noise_y]
               ])   # measurement covariance only apply on positions

buffer = 10
turning_sensitivity = 0.001


# update equation

H = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0, 0.0]])   # observation matrix

P = np.eye(5)  # initial covariance

    
    
        
class EKF:
    def __init__(self, H, x, P, Q, R):
        self.H = H 
        self.state_n = x
        self.covariance_n = P 
        self.error_move = Q 
        self.error_measurement = R 
        self.JA = None
        
    def predict(self, Zn, bug):    
                
        predicted_state = self.get_next_Xn(bug)
        JA = self.get_next_JA(predicted_state)
        self.JA = JA
        predicted_cov = self.JA * self.covariance_n * np.transpose(self.JA) + self.error_move
        observe_cov = self.H * predicted_cov * np.transpose(self.H) + self.error_measurement
        kalman_gain = predicted_cov * np.transpose(self.H) * np.linalg.inv(observe_cov)
        #print 'bug heading', bug.heading
        #print 'bug turn rate' , bug.turn_rate

        if Zn != 'NA':
            Zn = Zn.reshape(self.H.shape[0],1)
            observe = Zn - self.H * predicted_state
            self.state_n = predicted_state + kalman_gain * observe
        else:
            bug.v = predicted_state[3, 0]
            bug.set_heading((predicted_state[0, 0], predicted_state[1, 0]), True)
            #print 'turnrate', bug.turn_rate
            sign = bug.turn_rate > 0
            value = min(capped_tr, abs(bug.turn_rate))
            bug.turn_rate = value if sign else -value
            self.state_n = predicted_state
            
            
        dimesion = self.covariance_n.shape[0]
        self.covariance_n = (np.eye(dimesion)-kalman_gain * self.H) * predicted_cov
        
    def get_state_n(self):
        return self.state_n
    
    def get_next_Xn(self, bug):
        Xn = self.state_n
        if abs(bug.turn_rate) <= turning_sensitivity:
            Xn[0] = Xn[0] + Xn[3]*t * np.cos(Xn[2]) + 1/2*u*(t**2)   # Sx = Sx0+v0*t*cos(heading)+1/2ut^2
            Xn[1] = Xn[1] + Xn[3]*t * np.sin(Xn[2]) + 1/2*u*(t**2)
            Xn[3] += u*t
                                                                                                                                                                
        else:           
            Xn[0] = Xn[0] + Xn[3]*t * np.cos(bug.heading) + 1/2*u*(t**2)   
            Xn[1] = Xn[1] + Xn[3]*t * np.sin(bug.heading) + 1/2*u*(t**2)
            Xn[2] = bug.heading
            Xn[3] += u*t
            Xn[4] = bug.turn_rate
            
        return Xn
    
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

class HexBug:
    def __init__(self, wall, center, radius):
        self.wall = wall
        self.center = center
        self.radius = radius
        self.v = 0
        self.heading = 0
        self.turn_rate = 0
        self.damp = True
        
        
    def update_wall(self, x_min, x_max, y_min, y_max):
        self.wall = (x_min, x_max, y_min, y_max)

    def update_radius(self, closest):
        distance = self.distance_between(self.center, closest)
        self.radius = min(self.radius, distance)
        
    def distance_between(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def set_velocity(self, samples):
        # to do, wall bounce and corner stuck
        s = self.distance_between(samples[0], samples[1])
        self.v = s*1.0/t
    
    def set_heading(self, samples, predict_mode=False):
        if not predict_mode:
            self.heading = self.calculate_heading(samples)
            #print 'tracking heading', self.heading
            return
  
        heading = self.heading + self.turn_rate*t 
        if not self.damp:
            self.damp = self.damp_behavior(samples)
            
        #print '150', heading   
        heading = self.get_wall_reflection_ifneeded(samples, heading)
            #print '152', heading
        heading = self.get_center_reflection_ifneeded(samples, heading)
            #print '154', heading
        self.heading = heading
        
    def calculate_heading(self, samples):
        pt1, pt2 = samples
        try:
            heading = math.atan2(pt2[1]-pt1[1],pt2[0]-pt1[0])
        except Exception :
            heading = math.pi/2 if pt2[1]>pt1[1] else - math.pi/2
            
        return heading
    
    
    def set_turn_rate(self, samples):
        # this needs at least 3 points
        turn = self.get_turn(samples)     
        self.turn_rate = random.gauss(turn*1.0/2/t, 2*math.pi/3600)
        #print 'turn_rate', self.turn_rate
    
    def get_turn(self, samples):
        pt0, pt1, pt2 = samples
        turn1 = self.calculate_heading([pt0, pt1])
        turn2 = self.calculate_heading([pt1, pt2])
        turning_angle = turn1-turn2
        # check clockwise or counterclockwise turn        
        clockwise = turn1 > turn2
        turning_angle = -1*turning_angle if clockwise else turning_angle
        return turning_angle
    
        
    def is_hit_wall(self, position):
        x, y = position
        status_code = []
        if x <= self.wall[0] + buffer:
            status_code.append(1)
        elif x >= self.wall[1] - buffer:
            status_code.append(2)
        
        if y <= self.wall[2] + buffer:
            status_code.append(3)
        elif y >= self.wall[3] - buffer:
            status_code.append(4)

        return status_code
    
    def is_hit_center(self, position):
        return self.distance_between(position, center) <= self.radius
    
    def angle_trunc(self, a):
        """This maps all angles to a domain of [-pi, pi]"""
        while a < 0.0:
            a += math.pi * 2
        return ((a + math.pi) % (math.pi * 2)) - math.pi
    
    
    def get_wall_reflection_ifneeded(self, position, heading):
        
        status_code = self.is_hit_wall(position)
        if not status_code:
            return heading
        heading = self.angle_trunc(heading)
        if len(status_code) == 1:
            if status_code[0] == 1:
                return -1*heading if heading <math.pi/2 else math.pi-heading
            elif status_code[0] == 3:
                return math.pi - heading if heading > 0 else -heading
            elif status_code[0] == 2:
                return -heading if math.pi/-2 <heading < 0 else math.pi-heading
            else:
                return -heading  if math.pi>heading>math.pi/2 else math.pi-heading              
        elif len(status_code) == 2:
            return math.pi/2 -heading
        
        
    def get_center_reflection_ifneeded(self, position, heading):  
        pt1, pt2 = (0,0), (position[0]*math.cos(heading), position[1]*math.sin(heading))
        reflection = heading
        if self.is_hit_center(position):
            v = np.array([pt2[0]-pt1[0], pt2[1]-pt1[1]])
            n = np.array([pt2[0]-center[0], pt2[1]-center[1]])

            res = v - 2*(np.dot(v, n))*n
            reflection = self.calculate_heading([res.tolist(), (0, 0)])
        return reflection
        
    def damp_behavior(self, position):  
        x, y = position
        if x < self.wall[0] + 7*buffer:
            if y-self.wall[2]>self.wall[3]-y:
                self.turn_rate = capped_tr
            else:
                self.turn_rate = -capped_tr
            return True
        
        elif x > self.wall[1] - 7*buffer:
            print 'there'
            if y-self.wall[2]>self.wall[3]-y:
                self.turn_rate = -capped_tr
            else:
                self.turn_rate = capped_tr
            return True     
        if y < self.wall[2] + 7*buffer:
            if x-self.wall[0]> self.wall[1]-x:
                print 'here'
                self.turn_rate = -capped_tr
            else:
                self.turn_rate = capped_tr
            return True
        elif y < self.wall[3] - 7*buffer:
            if  x-self.wall[0]> self.wall[1]-x:
                self.turn_rate = capped_tr
            else:
                self.turn_rate = -capped_tr
            return True
        return False
    
    
def get_training_data(file_name):
    pts = []
    with open(file_name, 'r') as f:
        for line in f.readlines()[:sample_size]:
            pt = line.strip('\n').split(',')
            pts.append((int(pt[0]), int(pt[1])))
    return pts
    
    
def main(arg):
    
    file_name = 'inputs-txt/test02.txt'
    predicted_x = []
    predicted_y = []
    
    points = get_training_data(file_name)
    init_x, init_y = points[0][0], points[0][1]
    x = np.matrix([[init_x],     # x position
                   [init_y],     # y position
                   [0],          # heading
                   [0.1],         # velocity
                   [0]])         # turn rate
    
    kf = EKF(H, x, P, Q, R)
    x_min = x_max = init_x
    y_min = y_max = init_y
    bug = HexBug((x_min, x_max, y_min, y_max), center, radius)
    
    for i in xrange(3, sample_size):
        
        predicted_x.append(kf.get_state_n()[0, 0])
        predicted_y.append(kf.get_state_n()[1, 0])
                    
        if i >= training_size:
            Zn = 'NA'
        else:
            measured_x, measured_y = points[i][0], points[i][1]
            x_min = min(measured_x, x_min)
            x_max = max(measured_x, x_max)
            y_min = min(measured_y, y_min)
            y_max = max(measured_y, y_max)
        
            bug.update_wall(x_min, x_max, y_min, y_max)
            bug.update_radius(points[0])
            bug.set_velocity([points[i-1], points[i]])
            bug.set_heading([points[i-1], points[i]])
            bug.set_turn_rate(points[i-3:i])
            Zn = np.matrix([[measured_x],
                            [measured_y]])
       
        kf.predict(Zn, bug)
    
    with open('prediction.txt', 'w') as f:
        for x, y in zip(predicted_x[-60:], predicted_y[-60:]):
            f.write("%s,%s\n" %(int(x), int(y)))

    xs = [p[0] for p in points[:sample_size]]
    ys = [p[1] for p in points[:sample_size]]
    #for i in xrange(4, sample_size):
    #    print i, points[i], (predicted_x[i-4], predicted_y[i-4])
    pylab.plot(xs[-200:], ys[-200:],'-', predicted_x[-200:],predicted_y[-200:],'o')
    #pylab.plot(xs, ys,'-', predicted_x,predicted_y,'o')
    
    pylab.xlabel('X position')
    pylab.ylabel('Y position')
    pylab.title('Measurement of hexbug')
    pylab.legend(('measured','kalman'))
    pylab.show()   
    

if __name__ == "__main__":
    main(sys.argv[1:])
        
        