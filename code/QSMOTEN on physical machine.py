from pyezQ import * 
import numpy as np
from math import pi
import math

class Qgate():
    def __init__(self):
        self.fixed_X_gate = '''X Q%s
        '''
        self.fixed_H_gate = '''H Q%s
        '''
        self.fixed_X2M_gate = '''X2M Q%s
        '''
        self.fixed_X2P_gate = '''X2P Q%s
        '''
        self.fixed_Y2M_gate = '''Y2M Q%s
        '''
        self.fixed_Y2P_gate = '''Y2P Q%s
        '''
        self.fixed_RX_gate = '''RX Q%s %s
        '''
        self.fixed_RY_gate = '''RY Q%s %s
        '''
        self.fixed_RZ_gate = '''RZ Q%s %s
        '''
        self.fixed_CZ_gate = '''CZ Q%s Q%s
        '''
        self.fixed_measure_gate = '''M Q%s
        '''
        self.fixed_T_gate = '''T Q%s
        '''
        self.fixed_TD_gate = '''TD Q%s
        '''
    def ReArea(self, theta):
        theta = theta % (2*np.pi)
        if theta > np.pi:
            theta = theta - 2*np.pi
        return theta

    def X(self, bit):
        return self.fixed_X_gate % (bit)

    def H(self, bit):
        return self.fixed_H_gate % (bit)

    def X2M(self, bit):
        return self.fixed_X2M_gate % (bit)

    def X2P(self, bit):
        return self.fixed_X2P_gate % (bit)

    def SX(self, bit):
        return self.fixed_X2P_gate % (bit)

    def Y2M(self, bit):
        return self.fixed_Y2M_gate % (bit)

    def Y2P(self, bit):
        return self.fixed_Y2P_gate % (bit)

    def SY(self, bit):
        return self.fixed_Y2P_gate % (bit)

    def T(self, bit):
        return self.fixed_T_gate % (bit)

    def TD(self, bit):
        return self.fixed_TD_gate % (bit)

    def RX(self, theta, bit):
        theta = self.ReArea(theta)
        return self.fixed_RX_gate % (bit, theta)

    def RY(self, theta, bit):
        theta = self.ReArea(theta)
        return self.fixed_RY_gate % (bit, theta)

    def RZ(self, theta, bit):
        theta = self.ReArea(theta)
        return self.fixed_RZ_gate % (bit, theta)

    def P(self, theta, bit):
        theta = self.ReArea(theta)
        return self.fixed_RZ_gate % (bit, theta)

    def U(self, theta, phi, lamb, bit):
        cir = ''''''
        cir += self.P(phi + np.pi, bit)
        cir += self.SX(bit)
        cir += self.P(theta + np.pi, bit)
        cir += self.SX(bit)
        cir += self.P(lamb, bit)     
        return cir

    def CX(self, bit1, bit2):
        cir = ''''''
        cir += self.Y2M(bit2)
        #cir += self.H(bit2)
        cir += self.CZ(bit1, bit2)
        #cir += self.H(bit2)
        cir += self.Y2P(bit2)
        return cir

    def CZ(self, bit1, bit2):
        return self.fixed_CZ_gate % (bit1, bit2)

    def measure(self, bit):
        return self.fixed_measure_gate % (bit)

    def swap(self, bit1, bit2):
        cir = ''''''
        cir += self.CX(bit1, bit2)
        cir += self.CX(bit2, bit1)
        cir += self.CX(bit1, bit2)
        return cir

    def CX_swap(self, bit1, bit2, bit3):
        cir = ''''''
        cir += self.Y2M(bit2)
        cir += self.swap(bit2,bit3)
        cir += self.CZ(bit1,bit3)
        cir += self.swap(bit3,bit2)
        cir += self.Y2P(bit2)
        return cir


G = Qgate()

q0 = 31
q1 = 24
q2 = 30

cir = ''''''
sample1 = 0
sample2 = 0
if sample2 == 1:
	cir = cir + G.X(q1)

cir = cir + G.H(q2)

cir = cir + G.CX(q1,q0)
cir = cir + G.H(q1)
cir = cir + G.CX(q0,q1)
cir = cir + G.TD(q1)
cir = cir + G.CX(q2,q1)


cir = cir + G.T(q1)
cir = cir + G.CX(q0,q1)

cir = cir + G.T(q0)
cir = cir + G.TD(q1)
cir = cir + G.CX(q2,q1)


cir = cir + G.swap(q0,q1)

cir = cir + G.CX(q2,q1)
cir = cir + G.T(q2)
cir = cir + G.TD(q1)
cir = cir + G.CX(q2,q1)


cir = cir + G.H(q2)
cir = cir + G.measure(q2)


###The login_key can be access at: https://quantumcomputer.ac.cn/
account = Account(login_key='*********', machine_name='Xiaohong')


query_id = account.submit_job(cir, num_shots=10000)
outputstate = {}
if query_id:
    outputstate = account.query_experiment(query_id, max_wait_time=36000)
    if outputstate == {}:
        print(cir)


print(outputstate[0]["probability"])
p = outputstate[0]["probability"]['1']
print(p)
if p <= 0.5:
    sim =  math.sqrt(1 - 2*p)
else:
	sim = 0
print(sim)