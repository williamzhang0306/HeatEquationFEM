from math import sqrt,pi,exp,sin
import matplotlib.pyplot as plt
import numpy as np

def gauss_quadrature(f):
    '''Integrates f over the interval [-1,1] using 2nd order gauss quadrature'''
    return f(-1/sqrt(3)) + f(1/sqrt(3))

f = lambda x,t : (pi**2 - 1)*exp(-t)*sin(pi*x)
phi_1 = lambda chi: (1 - chi)/2

h=1/10
dx_dchi = h/2 
t_elapsed = 1/551
xi = 0
integrand = lambda chi: f((chi+1)*h/2 +xi, t_elapsed)*phi_1(chi)*dx_dchi
r = gauss_quadrature(integrand)
print(r)