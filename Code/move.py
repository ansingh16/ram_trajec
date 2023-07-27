#!/home/storage1/ph14036/miniconda2/bin/python
import Galaxy
import IGM
from ConfigParser import SafeConfigParser
from astropy import units as u
import numpy as np
from scipy import integrate

import sys

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
    if name=='n1':
        value = parser.getint('move', name)
    elif name=='n':
        value = parser.getint('move', name)
    elif name =='z':
        value = parser.getfloat('move', name)
    else:
        value = parser.get('move', name)

    para[name] = value
    #print '  %-12s : %-7r -> %0.2f' % (name, string_value, value)

print "n1,n = ",para['n1'],para['n']

gal = Galaxy.Galaxy(para)
IGM = IGM.Cluster(para)

r_c200 = IGM._R_200

r_box = 4.0*r_c200

print "r_box",r_box
v_c200 = IGM._V200

t_total = r_box/v_c200

v=v_c200*para['vel_factor']


print "total time taken to cross cluster: ",(r_box/v_c200).to('Myr')

dt = t_total/para['n1']
dr = gal._r_out/para['n']

#print "steps :",dt,dr


def acc(x1,v1,x2):
    A = (IGM.RAM_F(x1,v1,IGM) - gal.RES_F(x2))/(gal.sig_g(x2))
    #print IGM.RAM_F(x1,v1),gal.RES_F(x2),gal.sig_g(x2)
    return A

n = para['n']
n1= para['n1']

outfile = para['outfile']
fp3 = open(outfile,"w")



dm=0.0*u.Msun




for i in range(1,n-1):
    
    r_l =  gal._r_out - (i+1)*dr
    r_u =  gal._r_out - (i)*dr
    
    
    d_i = 0.0*u.Mpc
    d_f = 0.0*u.Mpc
    u_i = 0.0*u.Mpc/u.Myr
    u_f = 0.0*u.Mpc/u.Myr

    rc = r_c200

    for j in range(1,n1+1):



            #since we are starting from initial time equal to zero

            t_i = (j - 1)*dt
            t_f = j*dt


            #rbg should be in units of mpc which is why factor of thousand


            r_mid = (r_l + r_u) / 2.0

            #fp1.write("%f %f %f %f %f %f \n" % (r_mid.value, rc.value, IGM.RAM_F(rc,v).value, gal.RES_F(r_mid).value, v.value, IGM.rho_beta(rc).value))

            if (acc(rc,v,r_mid) > 0.0):
                # factor of 10^15 is to write in units of delta_d in units of KPC
                u_f = u_i + acc(abs(rc),v,r_mid) * dt
                delta_d = (u_i * dt + 0.50 * acc(abs(rc),v,r_mid) * (dt ** 2.0))
                d_f = d_i + delta_d
                u_i = u_f
                d_i = d_f

                # print acc(rc,r_mid),d_f,d_i

                #fp2.write("%f %f %f %f %f \n" % (delta_d.value, dt.value, acc(rc, v,r_mid).value, rc.value, r_mid.value))

                if ((gal._z_d - d_i).value > gal._eps):
                    continue
                else:
                    dm = dm + 2.0 * np.pi * r_mid * gal.sig_g(r_mid) * dr
                    #print r_mid/gal._r_dg,r_mid,rc/IGM._R_200
                    # print (IGM.rho_beta(rc)*v**2),2*np.pi*G*(gal._sigs0**2)*((u.Msun/u.Mpc**2)**2)*para['f_g']*(1.0 + para['f_g'])
                    #print r_mid / gal._r_dg, rc / IGM._R_200, r_mid.value, -(gal._r_ds / (2.0 * gal._r_dg)) * (np.log((IGM.rho_beta(rc,IGM,IGM._C) * v ** 2) / (2 * np.pi * G * (gal._sigs0 ** 2) * ((u.Msun / u.Mpc ** 2) ** 2) * para['f_g'] * (1.0 + para['f_g']))))

                    print r_mid / gal._r_dg,abs(rc) / IGM._R_200, r_mid.value, -(gal._r_ds/(gal._r_ds + gal._r_dg))*(np.log((IGM.rho_beta(abs(rc),IGM,IGM._C) * v ** 2) / (2 * np.pi * G * (gal._sigs0 *gal._sigg0)*((u.Msun / u.Mpc ** 2) ** 2))))

                    fp3.write("%f %f %f %f  \n" % (abs(rc)/IGM._R_200, r_mid/gal._r_dg,gal._r_out.to('kpc').value,r_mid.to('kpc').value))
                    break

            rc=rc-v*dt

#fp1.close()
#fp2.close()
fp3.close()
