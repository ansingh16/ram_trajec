import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from scipy.misc import derivative
from astropy.constants import G
from scipy import integrate

G = G.to('Mpc3/Msun Myr2')


f_g = 0.8
O_b = 0.015/(0.7**2)
O_m = 0.3


omega_m= 0.316
omega_l=  0.684
omega_k= 0.0
w0= -1.0
wa= 0.0
H0=0.00010227121650537077
f_uni= 0.158
h= 0.67

M_gal = 1.0e13*u.Msun
M_ICM = 2.0e15*u.Msun

z=0.0


def f_de(a):
    epsilon = 0.000000001
    return -3.0 * (1.0 + w0) + 3.0 * wa * ((a - 1.0) / np.log(a - epsilon) - 1.0)


def Esqr(a):
    return omega_m*pow(a, -3) + omega_k*pow(a, -2) + omega_l*pow(a, f_de(a))

def H(a):
    return h*H0 * np.sqrt(Esqr(a))






def f1(x,R_CORE,c1):
    beta=0.6
    FUNCT1 = (x**2)/((1+x**2)**(3.0*beta/2.0))
    return FUNCT1


def f2(x,R_vir,c1):
    c = c1
    FUNCT2 =  ((x+0.000001)**2)/((c*(x+0.000001))*((1.0 + c*(x+0.000001))**2))
    return FUNCT2


def M_NFW(r_c,R_200):
    H_z = H(1.0 / (1 + z)) * (1 / u.Myr)
    rho_c = 3.0 * H_z ** 2 / (8.0 * np.pi * (G.to('Mpc3/(Msun Myr2)')))
    c=10
    delta_c = (200.0 / 3.0) * (c ** 3 / (np.log(c + 1.0) - (c / (1 + c))))
    ymax = r_c/R_200
    R = R_200.value
    M = 4*np.pi*delta_c*rho_c*(R_200**3)*integrate.romberg(f2, 0.0, ymax.value,args=(R,c), divmax=100)

    print integrate.romberg(f2, 0.0, ymax.value,args=(R,c), divmax=100),rho_c*delta_c

    return M


def Beta_M(r_c,R_200):

    H_z = H(1.0/(1 + z))*(1 / u.Myr)
    c = 10.0


    f_uni=0.158
    rho_c = (3.0 * (H_z ** 2)) / (8.0 * np.pi * (G.to('Mpc3/(Msun Myr2)')))
    delta_c = (200.0 / 3.0) * ((c ** 3) / (np.log(c + 1.0) - (c / (1 + c))))

    R = R_200.value / 20.0
    R1 = R_200.value
    del1 = 1.0
    f_g = 0.8
    O_b = 0.022/(0.7**2)
    O_m = 0.316

    DELTA = f_g*O_b/O_m #f_uni/(1.0 - f_uni)#del1*f_uni


    M_BG = integrate.romberg(f2,0.0, 1.0,args=(R1,c),divmax=100)
    CONST = integrate.romberg(f1, 0.0, 20.0,args=(R,c),divmax=100)
    rho_0 = 0.5*DELTA*(20.0**3)*(M_BG/CONST)*(rho_c*delta_c)

    ymax = r_c /(R*u.Mpc)
    c1=10.0

    #print rho_c,delta_c

    M = 4.0*np.pi*((R_200/20.0)**3)*(rho_0)*integrate.romberg(f1, 0.0, ymax.value,args=(R,c1), divmax=100)

    return M







omega_m_z = (omega_m*((1.0+z)**3))/(omega_m*((1.0+z)**3) + omega_k*((1.0+z)**2) + omega_l)
d = omega_m_z - 1.0
delta_c = 18.0 * np.pi ** 2 + 82.0 * d - 39.0 * d ** 2

R_200 = (0.784*((M_ICM*h/(1.0e+8*u.Msun))**(1.0/3.0))*((omega_m_z*18.0*np.pi**2/(omega_m*delta_c))**(1.0/3.0))*(10.0/(h*(1.0+z)))*u.kpc).to('Mpc')

V200 = (23.4*((M_ICM*h / (1.0e8*u.Msun))**(1.0 / 3.0))*(((omega_m*delta_c) / (omega_m_z*18.0*(np.pi ** 2)))**(1.0 / 6.0))*(((1.0 + z) / 10) ** (1.0 / 2.0))*(u.km / u.second)).to('Mpc/Myr')

print R_200



r_start = 2*R_200.value
r_in = 2*R_200.value

#v=-np.sqrt((8*G.value*M_ICM.value)/(5.0*R_200.value))*u.Mpc/u.Myr
#v = -np.sqrt(((2 * G.value *(M_ICM.value))  * (r_in - R_200.value)) / (R_200.value * r_in)) * u.Mpc / u.Myr

v = -np.sqrt(2.0 * R_200.value * (r_in - r_start) / (r_start * r_in)) * V200


t_ff = (np.pi/2)*((r_start*u.Mpc)**(3.0/2.0))/np.sqrt(2*G*(M_ICM+M_gal))
t_end = 2*t_ff
dt = t_end/1000

a = -G*(M_NFW(r_start*u.Mpc,R_200)+Beta_M(r_start*u.Mpc,R_200))/((r_start*u.Mpc)**2)

#print a

dt_out = 0.01
t_out = dt_out

t=0.0*u.Myr


r1=[]
t1=[]
v1=[]
Mass=[]
Mass2=[]
Mass3=[]


#r1.append(r_start/R_200.value)
#t1.append(t.value/t_ff.value)


ekin1 = 0.5*M_gal*(v**2)
epot1 = -1.0*G*M_ICM*M_gal/np.sqrt((r_start*u.Mpc)**2)


e_in1 = ekin1 + epot1



#print rj.value,aj

r = r_start*u.Mpc

r2=[]
r3=[]
#print np.arange(0.0,R_200.value,R_200.value/100),R_200


for r0 in np.arange(0.001,R_200.value,R_200.value/500):
    Mass2.append(M_NFW(r0*u.Mpc,R_200).value)
    Mass3.append(Beta_M(r0*u.Mpc,R_200).value)
    r2.append(r0/R_200.value)
    r3.append(r0)

    #print r0/R_200.value,M_NFW(r0*u.Mpc,R_200),Beta_M(r0*u.Mpc,R_200)


#print len(Mass3),len(Mass2),len(r2)



Mass2 = np.array(Mass2)
Mass3 = np.array(Mass3)

M_tot = Mass3+Mass2

M_frac = Mass3/M_tot

print M_frac.mean(),Mass2[Mass2.shape[0]-1],Mass3[Mass3.shape[0]-1],f_g*O_b/O_m,Mass3[Mass3.shape[0]-1]/Mass2[Mass2.shape[0]-1]

plt.figure(1)
plt.loglog(r3,Mass2,ms=2,label='NFW')
plt.loglog(r3,Mass3,ms=2,label='Beta Model')
plt.xlabel(r"$R(Mpc)$")
plt.ylabel(r"log M")
plt.legend(loc=4)
plt.grid()
plt.savefig('fig1.png')

plt.figure(5)
plt.scatter(r3,M_frac,s=2)
plt.xlabel(r"$R(Mpc)$")
plt.ylabel(r"$M_{gas}/M_{tot}$")
plt.savefig('fig5.png')



#plt.show()


while(t<=t_end):

    #print r,M_NFW(r,R_200)

    #print r / R_200,R_200

    if (r > 0.0 * u.Mpc):

        #print r.value / R_200.value, v / V200, a,r_start,M_NFW(r,R_200),Beta_M(r,R_200),R_200**3

        r1.append(r.value / R_200.value)
        t1.append(t.value / t_ff.value)
        v1.append(v.to('km/s').value/V200.to('km/s').value)
        Mass.append(M_NFW(r,R_200).value)

        #print dt,a
        v += 0.5 * a * dt
        r += (v * dt)


        a = -G*(M_NFW(r,R_200)+Beta_M(r,R_200))/(r**2)#-derivative(Phi_dm, ri.value, dx=1.0e-3)*(u.Mpc/u.Myr**2)


        #print a
        v += 0.5 * a * dt






    else:

        ekin1 = 0.5 * M_gal * (v ** 2)
        epot1 = -1.0 * G *(M_NFW(r,R_200)+Beta_M(r,R_200))* M_gal / (np.sqrt(r ** 2))
        e_out1 = ekin1 + epot1

        ekin2 = 0.5 * M_gal * (v ** 2)
        epot2 = -1.0 * G * M_ICM * M_gal / np.sqrt(r ** 2)
        e_out2 = ekin2 + epot2
        '''
        print "Final total energy E_out = ", e_out1, e_out2
        print "absolute energy error: E_out - E_in = ", e_out1 - e_in1, e_out2 - e_in2
        print "relative energy error: (E_out - E_in) / E_in = ", (e_out1 - e_in1) / e_in1, (e_out2 - e_in2) / e_in2
        '''
        break

    t=t+dt




plt.figure(2)
plt.scatter(t1,r1,s=2)
plt.xlabel(r"$t/t_{ff}$")
plt.ylabel(r"$R/R_{200}$")
plt.savefig('fig2.png')



plt.figure(3)
plt.scatter(r1,np.abs(v1),s=2)
plt.xlabel(r"$R/R_{200}$")
plt.ylabel(r"$v/V_{200}$")
plt.savefig('fig3.png')



plt.figure(4)
plt.scatter(r1,np.log10(Mass),s=2)
plt.xlabel(r"$R/R_{200}$")
plt.ylabel(r"log M")
plt.savefig('fig4.png')



plt.show()


