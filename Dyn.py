import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from scipy.misc import derivative
from astropy.constants import G
from scipy import integrate
import cosmology


class VariableV:


    def M_NFW(self,r_c,R_200,IGM,para):

                cos = cosmology.Cosmology(para)

                H_z = cos.H(1.0 / (1 + cos._z)) * (1 / u.Myr)

                rho_c = 3.0 * H_z ** 2 / (8.0 * np.pi * (G.to('Mpc3/(Msun Myr2)')))

                delta_c = (200.0 / 3.0) * (IGM._C ** 3 / (np.log(IGM._C + 1.0) - (IGM._C / (1 + IGM._C))))

                ymax = r_c/R_200
                R = R_200.value
                M = 4*np.pi*delta_c*rho_c*(R_200**3)*integrate.romberg(IGM.f2, 0.0, ymax.value,args=(R_200.value,IGM._C), divmax=100)
                #print integrate.romberg(IGM.f2, 0.0, ymax.value,args=(R_200.value,IGM._C), divmax=100)
                return M


    def Beta_M(self,r_c,R_200,IGM):
                R = IGM._R_CORE.value
                M_BG = integrate.romberg(IGM.f2,0.0, 1.0,args=(R_200.value,IGM._C),divmax=100)
                CONST = integrate.romberg(IGM.f1, 0.0, 1*IGM._R_200.value/IGM._R_CORE.value,args=(R_200.value,IGM._C),divmax=100)
                rho_0 = 0.5*(IGM._rho_c * IGM._delta_c) * IGM._DELTA * (20.0 ** 3) * M_BG / CONST
                ymax = r_c /(R*u.Mpc)
                M = 4.0*np.pi*((R_200/20.0)**3)*(rho_0)*integrate.romberg(IGM.f1, 0.0, ymax.value,args=(R_200.value,IGM._C) ,divmax=100)
                return M

    def dist(self,a, b, c, d):
                 return np.sqrt((a - b) ** 2 + (c - d) ** 2)

    def vel_provider(self,x1,x2,y1,y2,vx1,vx2,vy1,vy2,ax1,ax2,ay1,ay2,dt,IGM,Gal,para):

        #if (R > 0.0 * u.Mpc):

                from astropy.constants import G  #print dt,a

                G = G.to('Mpc3/(Msun Myr2)')

                #print ax1,dt,vx1

                vx1 = (vx1.value + 0.5*ax1.value*dt.value)*u.Mpc/u.Myr
                #vx2 = (vx2.value + 0.5*ax2.value * dt.value)*u.Mpc/u.Myr
                vy1 = (vy1.value + 0.5 * ay1.value * dt.value)*u.Mpc/u.Myr
                #vy2 = (vy2.value + 0.5 * ay2.value * dt.value)*u.Mpc/u.Myr
            
                x1 = x1 + dt * vx1
                y1 = y1 + dt * vy1
                #x2 = x2 + dt * vx2
                #y2 = y2 + dt * vy2


                M_r = self.M_NFW(self.dist(x1, x2, y1, y2), IGM._R_200,IGM,para) + self.Beta_M(self.dist(x1, x2, y1, y2), IGM._R_200,IGM)
            
                ax1 = (G * M_r / (self.dist(x1, x2, y1, y2) ** 2)) * ((x2 - x1) / (self.dist(x1, x2, y1, y2)))
                ay1 = (G * M_r / (self.dist(x1, x2, y1, y2) ** 2)) * ((y2 - y1) / (self.dist(x1, x2, y1, y2)))
                #ax2 = (G * Gal._md_tot / (self.dist(x1, x2, y1, y2) ** 2)) * ((x1 - x2) / (self.dist(x1, x2, y1, y2)))
                #ay2 = (G * Gal._md_tot / (self.dist(x1, x2, y1, y2) ** 2)) * ((y1 - y2) / (self.dist(x1, x2, y1, y2)))

                vx1 = (vx1.value + 0.5 * ax1.value * dt.value) * u.Mpc / u.Myr
                #vx2 = (vx2.value + 0.5 * ax2.value * dt.value) * u.Mpc / u.Myr
                vy1 = (vy1.value + 0.5 * ay1.value * dt.value) * u.Mpc / u.Myr
                #vy2 = (vy2.value + 0.5 * ay2.value * dt.value) * u.Mpc / u.Myr

                return x1,x2,y1,y2,vx1,vx2,vy1,vy2,ax1,ax2,ay1,ay2




