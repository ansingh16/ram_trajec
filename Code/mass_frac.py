import numpy as np
import os

import sys
import Galaxy
import IGM
from ConfigParser import SafeConfigParser
from astropy import units as u
import numpy as np
from scipy import integrate

parser = SafeConfigParser()

for dirpath, dirnames, filenames in os.walk("."):
    for filename in [f for f in filenames if f.endswith(".ini")]:

        full_path = os.path.join(dirpath, filename)

        print full_path

        parser = SafeConfigParser()
        if(full_path=='./params.ini'):
            continue
        else:
            parser.read(full_path)

            para = {}

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
                else:
                    value = parser.get('move', name)

                para[name] = value
                # print '  %-12s : %-7r -> %0.2f' % (name, string_value, value)

            print "n1,n = ", para['n1'], para['n']

            gal1 = Galaxy.Galaxy(para)
            IGM1 = IGM.Cluster(para)

            r_c200 = IGM1._R_200

            r_box = 4.0 * r_c200

            print "r_box", r_box
            v_c200 = IGM1._V200

            t_total = r_box / v_c200

            v = v_c200 * para['vel_factor']

            print "total time taken to cross cluster: ", (r_box / v_c200).to('Myr')

            dt = t_total / para['n1']
            dr = gal1._r_out / para['n']

            # print "steps :",dt,dr
            n = para['n']
            n1 = para['n1']


            data = np.genfromtxt(full_path.replace('params.ini','.dat'))


            print data.shape

            m1_total=0.0
            for ii in range(1,data.shape[0]):
                dm = 2.0*np.pi*data[ii,3]
                m1_total = m1_total + dm

            M_frac = m1_total/dm




