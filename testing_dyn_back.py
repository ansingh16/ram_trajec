
import Galaxy
import IGM
from ConfigParser import SafeConfigParser
from astropy import units as u
import numpy as np
from scipy import integrate
import Dyn
import sys
import matplotlib.pyplot as plt
from matplotlib import animation

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


parser = SafeConfigParser()
parser.read(sys.argv[1])

para = {}

from astropy.constants import G

G = G.to('Mpc3/(Msun Myr2)')

print '\nCosmology:'
for name in parser.options('cosmology'):
    string_value = parser.get('cosmology', name)
    value = parser.getfloat('cosmology', name)
    para[name] = value
    print '  %-12s : %-7r -> %0.2f' % (name, string_value, value)

print '\nCluster'
for name in parser.options('cluster'):
    string_value = parser.get('cluster', name)
    value = parser.getfloat('cluster', name)
    para[name] = value
    print '  %-12s : %-7r -> %0.2f' % (name, string_value, value)

print '\ngalaxy'
for name in parser.options('galaxy'):
    string_value = parser.get('galaxy', name)
    value = parser.getfloat('galaxy', name)
    para[name] = value
    print '  %-12s : %-7r -> %0.2f' % (name, string_value, value)

print '\nmove'
for name in parser.options('move'):
    string_value = parser.get('move', name)
    if name == 'n1':
        value = parser.getint('move', name)
    elif name == 'n':
        value = parser.getint('move', name)
    elif name == 'z':
        value = parser.getfloat('move', name)
    elif name == 'theta':
        value = parser.getfloat('move', name)
    elif name == 'vel_factor':
        value = parser.getfloat('move', name)
    else:
        value = parser.get('move', name)

    para[name] = value
    # print '  %-12s : %-7r -> %0.2f' % (name, string_value, value)

print "n1,n = ", para['n1'], para['n']

gal = Galaxy.Galaxy(para)
IGM = IGM.Cluster(para)
dyn = Dyn.VariableV()


r_c200 = IGM._R_200

r_box = 4.0 * r_c200

print "r_box", r_box
v_c200 = IGM._V200


r_in = 2*IGM._R_200.value


r_start = 2.0*IGM._R_200.value

v = -np.sqrt(2.0 * IGM._R_200.value * (r_in - r_start) / (r_start * r_in)) * IGM._V200

rc=r_start*u.Mpc


#a = -G *(dyn.M_NFW(rc, IGM._R_200,IGM)+dyn.Beta_M(rc,IGM._R_200,IGM)) / ((rc) ** 2)
a = -G * (dyn.M_NFW(r_start * u.Mpc, IGM._R_200, IGM,para) + dyn.Beta_M(r_start * u.Mpc, IGM._R_200, IGM)) / ((r_start * u.Mpc) ** 2)

t_ff = (np.pi/2)*((r_start*u.Mpc)**(3.0/2.0))/np.sqrt(2*G*(IGM._M_200+gal._md_tot))
t_total = t_ff
dt = t_total/para['n1']


print "total time taken to cross cluster: ", (r_box / v_c200).to('Myr')




#dt = t_total / para['n1']
dr = gal._r_out / para['n']


# print "steps :",dt,dr


def acc(x1, v1, x2):
    A = (IGM.RAM_F(x1, v1,IGM) - gal.RES_F(x2)) / (gal.sig_g(x2))
    # print IGM.RAM_F(x1,v1),gal.RES_F(x2),gal.sig_g(x2)
    return A


n = para['n']
n1 = para['n1']

outfile = para['outfile']

fp3 = open(outfile, "w")

fp4 = open('surface.txt','w')

dm = 0.0 * u.Msun

fig, ax = plt.subplots(figsize=(6, 6))
ax.grid()
time_text = ax.text(0.95, 0.95, '', horizontalalignment='right', verticalalignment='top',
                              transform=ax.transAxes)
xdata, ydata = [], []
# line, = ax.plot([], [], lw=2.0,label=r'$\theta = \pi/4, V_{in} = 1 \times V_{200}$')

scat, = ax.plot([], [], linestyle='', ms=2, marker='o', color='b',label=r'$\theta$ = '+str(para['theta'])+ r'$^{\circ}$'+', $V_{in}$ ='+str(para['vel_factor'])+ r'$\times V_{200}$')


time_text = ax.text(0.05, 0.02, '', horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

#for i in range(1, n-1):

dm=0.0


theta = para['theta']*np.pi/180.0
v_0 = -para['vel_factor'] * IGM._V200

#print "THETA IS : ", theta, "VELOCITY IS: ",v_0/IGM._V200


for i in range(1,n-1):



    fp3 = open(outfile, "a")

    r_l = gal._r_out - (i + 1) * dr
    r_u = gal._r_out - (i) * dr

    d_i = 0.0 * u.Mpc
    d_f = 0.0 * u.Mpc
    u_i = 0.0 * u.Mpc / u.Myr
    u_f = 0.0 * u.Mpc / u.Myr


    # setting up the initial conditions
    r_0 = 2 * IGM._R_200.value

    xgal = r_0 * np.cos(theta)*u.Mpc  # -400.011995319#
    ygal = r_0 * np.sin(theta)*u.Mpc  # 363.865867513#

    xclus = 0.0*u.Mpc  # 0.627784919989#
    yclus = 0.0*u.Mpc  # -0.571056636293#

    vxgal = v_0  # 86.5591029587#
    vygal = 0.0*u.Mpc/u.Myr  # v_0.value  # -28.7996765414#

    vxclus = 0.0*u.Mpc/u.Myr  # -0.135847174988#
    vyclus = 0.0*u.Mpc/u.Myr  # 0.0451986511525#


    v = np.sqrt(vxgal.value**2 + vygal.value**2)*u.Mpc/u.Myr


    axgal = (G * IGM._M_200 / (dyn.dist(xgal, xclus, ygal, yclus) ** 2))*((xclus - xgal) / (dyn.dist(xgal, xclus, ygal, yclus)))

    aygal = (G * IGM._M_200 / (dyn.dist(xgal, xclus, ygal, yclus) ** 2))*((yclus - ygal) / (dyn.dist(xgal, xclus, ygal, yclus)))

    axclus = (G * gal._md_tot / (dyn.dist(xgal, xclus, ygal, yclus) ** 2))*((xgal - xclus) / (dyn.dist(xgal, xclus, ygal, yclus)))

    ayclus = (G * gal._md_tot / (dyn.dist(xgal, xclus, ygal, yclus) ** 2))*((ygal - yclus) / (dyn.dist(xgal, xclus, ygal, yclus)))


    #print "M here",v/IGM._V200

    rc = dyn.dist(xgal, xclus, ygal, yclus)


    for j in range(1, n1 + 1):

        print rc,vxgal.value/IGM._V200.value


        if (rc.value > 0.0):

            t_i = (j - 1) * dt
            t_f = j * dt

            #print rc.value,vxgal,vygal

            #print rc / IGM._R_200,v / IGM._V200,a,r_start,dyn.M_NFW(rc, IGM._R_200,IGM,para),dyn.Beta_M(rc, IGM._R_200,IGM),

            # rbg should be in units of mpc which is why factor of thousand

            #print rc / IGM._R_200, v / IGM._V200

            r_mid = (r_l + r_u) / 2.0


            if (acc(rc, v, r_mid) > 0.0):

                #print "M here"
                # factor of 10^15 is to write in units of delta_d in units of KPC
                u_f = u_i + acc(rc, v, r_mid) * dt
                delta_d = (u_i * dt + 0.50 * acc(rc, v, r_mid) * (dt ** 2.0))
                d_f = d_i + delta_d
                u_i = u_f
                d_i = d_f

                if ((gal._z_d - d_i).value > gal._eps):
                    continue
                else:
                    #dm = dm + 2.0 * np.pi * r_mid * gal.sig_g(r_mid) * dr
                    # print r_mid/gal._r_dg,r_mid,rc/IGM._R_200
                    # print (IGM.rho_beta(rc)*v**2),2*np.pi*G*(gal._sigs0**2)*((u.Msun/u.Mpc**2)**2)*para['f_g']*(1.0 + para['f_g'])
                    #print r_mid / gal._r_dg, rc / IGM._R_200, r_mid.value, -(gal._r_ds / (2.0 * gal._r_dg)) * (np.log((IGM.rho_beta(rc,IGM,IGM._C) * v ** 2) / (2 * np.pi * G * (gal._sigs0 ** 2) * ((u.Msun / u.Mpc ** 2) ** 2) * para['f_g']*(1.0 + para['f_g']))))

                    #print IGM._V200,np.sqrt(G*IGM._M_200/IGM._R_200),v

                    #print r_mid / gal._r_dg, rc / IGM._R_200,v/IGM._V200

                    print r_mid / gal._r_dg,rc / IGM._R_200, -(gal._r_ds/(gal._r_ds + gal._r_dg))*(np.log((IGM.rho_beta(rc,IGM,IGM._C) * v ** 2) / (2 * np.pi * G * (gal._sigs0 *gal._sigg0)*((u.Msun / u.Mpc ** 2) ** 2)))),gal.sig_g(r_mid).value


                    #print np.sqrt(G*IGM._M_200/IGM._R_200)/IGM._V200,v.value/IGM._V200.value,r_start/IGM._R_200.value,r_in/IGM._R_200.value
                    #fp4.write("%f %f %f %f %f %f %f \n "%(r_mid.value, gal.sig_g(r_mid).value,gal.sig_s(r_mid).value,gal.RES_F(r_mid).value,rc.value,IGM.RAM_F(rc,v,IGM).value,acc(rc,v,r_mid).value))

                    fp3.write("%f %f %f %f %f \n" % (rc / IGM._R_200, r_mid / gal._r_dg, gal._r_out.to('kpc').value, r_mid.to('kpc').value,-(gal._r_ds/(gal._r_ds + gal._r_dg))*(np.log((IGM.rho_beta(rc,IGM,IGM._C) * v ** 2) /(2 * np.pi * G * (gal._sigs0 *gal._sigg0)*((u.Msun / u.Mpc ** 2) ** 2))))))
                    #gal._r_out= r_l
                    break


            xgal, xclus, ygal, yclus, vxgal, vxclus, vygal, vyclus, axgal, axclus, aygal, ayclus = dyn.vel_provider(xgal,xclus,ygal,yclus,vxgal,vxclus,vygal,vyclus,axgal,axclus,aygal,ayclus,dt,IGM,gal,para)
            rc = dyn.dist(xgal, xclus, ygal, yclus)
            #print r_c
            #print i, j,gal._r_out/gal._r_dg




# fp1.close()
# fp2.close()
fp3.close()
fp4.close()
#plt.show()