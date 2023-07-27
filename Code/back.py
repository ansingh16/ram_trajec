import numpy as np
from scipy import integrate



def funct_sig_g(x):
    f1 = x*np.exp(-1.0*x/r_dg)
    return f1

def funct_sig_s(x):
    f2 = x*np.exp(-1.0*x/r_ds)
    return f2

def sig_g(x):
    m_d1 = f_g*m_d
    r_max = 20.0*r_dg
    const = integrate.romberg(funct_sig_g, 0.0,r_max,divmax = 10)
    sig0 = m_d1/(2.0*np.pi*const)
    Sig_g = sig0*np.exp(-1.0*x/r_dg)
    return Sig_g

def sig_s(x):
    m_d1 = (1.0 - f_g)*m_d
    r_max = 20.0*r_ds
    const = integrate.romberg(funct_sig_s,0.0,r_max,divmax = 10)
    sig0 = m_d1/(2.0*np.pi*const)
    Sig_s = sig0*np.exp(-1.0*x/r_ds)
    return Sig_s


def rho_bg(x):
     dummy = 0.001
     rho_c = (3.0/(800.0*pi))*(M_200/R_200**3)*0.001
     delta_c = (200.0/3.0)*(C**3/(np.log(C+1.0) - (C/(1 + C))))
     x_0 = C*(x+dummy)/R_200
     RHO_BG = rho_c*delta_c/(x_0*(1.0 + x_0)**2)
     return RHO_BG 

def f1(x):
    x1 = x/R_CORE
    FUNCT1 = x*x*(1+x**2)**(-3.0*BETA/2.0)
    return FUNCT1

def f2(x):
    dummy = 0.0001
    rho_c = (3.0/(800.0*np.pi))*(M_200/R_200**3)*0.001
    delta_c = (200.0/3.0)*(C**3/(np.log(C+1.0) - (C/(1 + C))))
    R1_200 = R_200/R_CORE
    x_0 = C*(x+dummy)
    x1 = x/R_CORE
    FUNCT2 =  x*x*rho_c*delta_c/(x_0*(1.0 + x)**2)
    return FUNCT2


def rho_beta(x):

    M_BG = integrate.romberg(f2,R_INNER,1.0,divmax = 100)
    CONST = integrate.romberg(f1,R_INNER,20.0,divmax = 100)

    rho_0 = DELTA*(20.0**3)*M_BG/CONST
    x0 = x/R_CORE
    rho_beta = (rho_0*(1+x0**2)**(-3.0*BETA/2.0))
    return rho_beta


def rho_01(x):
    M_BG = integrate.romberg(f2,R_INNER,1.0,divmax = 10)
    CONST = integrate.romberg(f1,R_INNER,20.0,divmax = 10)
    Rho_0 = DELTA*(20.0**3)*M_BG/CONST
    return Rho_0

def RES_F(x2):

    f_s = 2.0*np.pi*(g)*sig_g(x2)*sig_s(x2)
    f_b = (g)*m_tot*sig_g(x2)/(x2**2.0)
    Res_f = f_s + epsilon*f_b
    return Res_f  

def RAM_F(x1):
    Ram_f = rho_beta(x1)*v*v/(1.0e-15)
    return Ram_f

 
def acc(x1,x2):
    A = (RAM_F(x1) - RES_F(x2))/(sig_g(x2)) # MY CHANGE
    return A


n = 100
n1= 100


#Universal baryonic fraction and redshift..

f_uni = 0.158
z = 0.0

#Disc parameters of Milky way...

M1 = 1.0e+12
R1 = 2.0e+4  

#Coma cluster parameters...

R2 = 3.0               # IN UNITS OF MPC
M2 = 2.0                # IN UNITS OF 10^15 SOLAR MASS
V1 = 0.0020438            # 2000 km/sec in units of pc/yr
'''
**************************************************************************
*     FOR THE SURFACE DENSITY OF DISC GAS AND STARS...
**************************************************************************
'''
alpha = 0.1
f_g = 0.5
md_tot = 1.0e+12
m_d = alpha*f_uni*md_tot

print 'Total mass is galaxy is', md_tot

lmbda = 0.04
omega_m = 0.3
omega_l = 0.7
omega_k = 0.0


omega_m_z = omega_m*(1.0+z)**3/(omega_m*(1.0+z)**3 + omega_l + omega_k*(1.0+z)**2)
d = omega_m_z - 1.0
delta_c = 18.0*np.pi**2 + 82.0*d - 39.0*d**2
h = 0.73
r_dg = (lmbda/np.sqrt(2.0))*0.784*((md_tot*h/1.0e+8)**(1.0/3.0))*((omega_m_z*18.0*np.pi**2/(omega_m*delta_c))**(1.0/3.0))*(10.0/(h*(1.0+z)))*1.0e+3
#factor of 10^3 is to convert in parsec
r_out = 10.0*r_dg
r_ds = 3.0*r_out/20.0


print 'r_dg is = ',r_dg
print 'r_out is= ',r_out
print 'Omega_m_z is = ',omega_m_z
print 'delta_c is= ',delta_c
'''
**************************************************************************
*     FOR NFW PROFILE....
**************************************************************************
'''
M_200 = 2.0             # IN UNITS OF 10^15 SOLAR MASS
R_200 = 0.784*((M_200*h*1.0e+7)**(1.0/3.0))*((omega_m_z*18.0*np.pi**2/(omega_m*delta_c))**(1.0/3.0))*(10.0/(h*(1.0+z)))*1.0e-3

#10^7 is coming as M_200 is written units of 10^15 solar mass
#10^-3 converts IN UNITS OF MPC
      
C = 10.0
'''
**************************************************************************
*     FOR BETA MODEL...
**************************************************************************
'''

BETA = 0.6
del1 = 1.0
DELTA = del1*f_uni
R_INNER = 0.0
R_CORE = R_200/(20.0*(1.0+z))            #IN UNITS OF MPC
R_VIRIAL = R_200/R_CORE
'''
*************************************************************************
*     GLOBAL PAREMETERS
*************************************************************************
'''    

z_d = 0.3               #IN UNITS OF KPC
r_box = 4.0*R_200       # IN UNITS OF MPC, USE EVEN MUTIPLES OF VIRIAL RADIUS..

v = np.sqrt(1.0 + z)*V1*((M_200/M2)**(1.0/3.0))

'''
wacky_unit_system = yt.UnitSystem("wacky", "pc", "Msun", "yr", temperature_unit="K", angle_unit="deg")
unit_system = yt.unit_system_registry["wacky"]
g= unit_system.constants.G.in_base('wacky')

print "G is:",g
'''
g = 4.49                #GRAVITATIONAL CONSTANT IN UNITS OF PC,YEAR,SOLAR MASS MULTIPLIED BY 10^15


m_bh = 4.0e+6
m_bul = 1.0e+9
m_tot = m_bh + m_bul
epsilon = 0.0           #ONE if central region is to be included else ZERO....
eps = 0.0001            # limit for difference in distance travelled by mass element and z_d
      
t_total = (r_box/v)*0.001 #This will give time in gigayears...

dt = t_total/n1     
dr = r_out/n


print 'v is ',v
print 'r_box is = ',r_box
print 'r_out is = ',r_out
print 't_total is =  ',t_total
print 'M200 is = ',M_200
print 'Beta is = ',BETA
print 'Velocity is =',v
print 'r_box is',r_box
print 'r_out is',r_out
print 'Total time(t_total) is',t_total
print 'Background mass is',M_200
print 'beta is',BETA
print 'disc mass is',md_tot
print 'redshift is',z


fp1 = open('out1.txt',"w")
fp2 = open('out2.txt',"w")
fp3 = open('out3.txt',"w")
fp4 = open('out4.txt',"w")



for i in range(1,n-1):
    
    r_l =  r_out - (i+1)*dr
    r_u =  r_out - (i)*dr
    
    '''
    ************************************************************************************
    *     Initializing distance and velocity...
    ************************************************************************************
    '''
    d_i = 0.0
    d_f = 0.0
    u_i = 0.0
    u_f = 0.0

   

    for j in range(1,n1+1):
        
            #since we are starting from initial time equal to zero
            
            t_i = (j - 1)*dt
            t_f = j*dt
            print j

            #rbg should be in units of mpc which is why factor of thousand
            
            rbg_i = (v*t_i)*1000.0
            rbg_f = (v*t_f)*1000.0 
            
            #this conditions converts rbg in terms of radius from center of halo
            
            if (rbg_i <= (r_box/2.0)):
                rc_i = r_box/2.0 - rbg_i
            else:
                rc_i = rbg_i - r_box/2.0
                
                
            if (rbg_f<=(r_box/2.0)):
                rc_f = r_box/2.0 - rbg_f
            else:
                rc_f = rbg_f - r_box/2.0
					
                
            rc = (rc_i + rc_f)/2.0          #in Mpc
            r_mid = (r_l + r_u)/2.0	    #in pc
            
            fp1.write("%f %f %f %f %f %f %f \n"%(r_mid,rc,RAM_F(rc),RES_F(r_mid),rho_01(rc),v,rho_beta(rc)))
            
            
            if (acc(rc,r_mid)>0.0): 
                
                #factor of 10^15 is to write in units of delta_d in units of KPC               
                u_f = u_i + acc(rc,r_mid)*dt
                delta_d = (u_i*dt + 0.50*acc(rc_f,r_mid)*(dt**2.0))
                d_f = d_i + delta_d
                u_i = u_f
                d_i = d_f

                #print acc(rc,r_mid),d_f,d_i
                
                fp2.write("%f %f %f %f %f \n"%(delta_d,dt,acc(rc,r_mid),rc,r_mid))
                
                if ((z_d - d_i)>eps): 
                    continue
                else:
                    dm = 2.0*np.pi*r_mid*sig_g(r_mid)*dr
                    fp3.write("%f %f %f %f %f %f %f \n" %(rc,r_mid,r_l,r_u,t_f,dm,z_d - d_i))
                    break
                    
					   
            else:
                
                fp4.write("%d %d\n" %(i,j))

           

       




      
fp1.close()
fp2.close()
fp3.close()
fp4.close()
      
      
