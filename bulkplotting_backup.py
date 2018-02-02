from matplotlib import rc,rcParams
rc('font', family='serif', serif='cm10')
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
rcParams['text.latex.preamble'] = [r'\boldmath']

import os
import os.path
import re
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from matplotlib.mlab import griddata
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
#plt.style.use('seaborn-talk')

matplotlib.rc('text', usetex=True)
matplotlib.rc('axes', linewidth=2)
matplotlib.rc('font', weight='bold')

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

import pandas as pd


def plot_styling(bx,flag):
    # ax.plot(data[:, 0], data[:, 1], col[i] + '.', ms=8.0, label=lmbda)

    if(flag=='rstrip'):

            bx.set_ylabel(r"$\mathbf{r_{strip}/r_{dg}}$", fontweight='bold', fontsize=16)
            bx.set_xlabel(r"$\mathbf{R/R_{200}}$", fontweight='bold', fontsize=16)
            bx.minorticks_on()
            bx.tick_params(axis='both', which='minor', length=3, width=2, labelsize=15)
            bx.tick_params(axis='both', which='major', length=5, width=2, labelsize=15)

    if (flag == 'M_removed'):

            bx.set_ylabel(r"$\mathbf{M_{removed}}$", fontweight='bold', fontsize=16)
            bx.set_xlabel(r"$\mathbf{R/R_{200}}$", fontweight='bold', fontsize=16)
            bx.minorticks_on()
            bx.tick_params(axis='both', which='minor', length=3, width=2, labelsize=15)
            bx.tick_params(axis='both', which='major', length=5, width=2, labelsize=15)

    return bx



def different_lambda_plotter(cx,dat,flag):

        if(flag=='M_removed'):

            cx[0].plot(dat[0].lambda002z00R_c, dat20.lambda002z00M_removed,'-',color='k')
            cx[0].plot(dat[1].lambda004z00R_c, dat40.lambda004z00M_removed,'--',color='k')
            cx[0].plot(dat[2].lambda008z00R_c, dat80.lambda008z00M_removed,'-.',color='k')
            cx[0].set_title('z = ' + '0.0', fontweight='bold')

            cx[1].plot(dat[3].lambda002z05R_c, dat21.lambda002z05M_removed,'-',color='k')
            cx[1].plot(dat[4].lambda004z05R_c, dat41.lambda004z05M_removed,'--',color='k')
            cx[1].plot(dat[5].lambda008z05R_c, dat81.lambda008z05M_removed,'-.',color='k')
            cx[1].set_title('z = ' + '0.5', fontweight='bold')

            cx[2].plot(dat[6].lambda002z10R_c, dat22.lambda002z10M_removed,'-',color='k',label=r'$\lambda = 0.02$')
            cx[2].plot(dat[7].lambda004z10R_c, dat42.lambda004z10M_removed,'--',color='k',label=r'$\lambda = 0.04$')
            cx[2].plot(dat[8].lambda008z10R_c, dat82.lambda008z10M_removed,'-.',color='k',label=r'$\lambda = 0.08$')
            cx[2].set_title('z = ' + '1.0', fontweight='bold')

            cx[2].legend(loc=1)

            for i in iter(cx):
                plot_styling(i, 'M_removed')


        if(flag=='r_strip'):

            cx[0].plot(dat[0].lambda002z00R_c, dat20.lambda002z00r_strip_r_dg,'-',color='k')
            cx[0].plot(dat[1].lambda004z00R_c, dat40.lambda004z00r_strip_r_dg,'--',color='k')
            cx[0].plot(dat[2].lambda008z00R_c, dat80.lambda008z00r_strip_r_dg,'-.',color='k')
            cx[0].set_title('z = ' + '0.0', fontweight='bold')

            cx[1].plot(dat[3].lambda002z05R_c, dat21.lambda002z05r_strip_r_dg,'-',color='k')
            cx[1].plot(dat[4].lambda004z05R_c, dat41.lambda004z05r_strip_r_dg,'--',color='k')
            cx[1].plot(dat[5].lambda008z05R_c, dat81.lambda008z05r_strip_r_dg,'-.',color='k')
            cx[1].set_title('z = ' + '0.5', fontweight='bold')

            cx[2].plot(dat[6].lambda002z10R_c, dat22.lambda002z10r_strip_r_dg,'-',color='k',label=r'$\lambda = 0.02$')
            cx[2].plot(dat[7].lambda004z10R_c, dat42.lambda004z10r_strip_r_dg,'--',color='k',label=r'$\lambda = 0.04$')
            cx[2].plot(dat[8].lambda008z10R_c, dat82.lambda008z10r_strip_r_dg,'-.',color='k',label=r'$\lambda = 0.08$')
            cx[2].set_title('z = ' + '1.0', fontweight='bold')

            cx[2].legend(loc=2)


            for i in iter(cx):
                plot_styling(i, 'rstrip')


def different_theta_plotter(cx,dat,flag):
    if (flag == 'M_removed'):

        cx[0].plot(dat[0].lambda002z00R_c, dat20.lambda002z00M_removed, '-', color='k')
        cx[0].plot(dat[1].lambda004z00R_c, dat40.lambda004z00M_removed, '--', color='k')
        cx[0].plot(dat[2].lambda008z00R_c, dat80.lambda008z00M_removed, '-.', color='k')
        cx[0].set_title('z = ' + '0.0', fontweight='bold')

        cx[1].plot(dat[3].lambda002z05R_c, dat21.lambda002z05M_removed, '-', color='k')
        cx[1].plot(dat[4].lambda004z05R_c, dat41.lambda004z05M_removed, '--', color='k')
        cx[1].plot(dat[5].lambda008z05R_c, dat81.lambda008z05M_removed, '-.', color='k')
        cx[1].set_title('z = ' + '0.5', fontweight='bold')

        cx[2].plot(dat[6].lambda002z10R_c, dat22.lambda002z10M_removed, '-', color='k', label=r'$\lambda = 0.02$')
        cx[2].plot(dat[7].lambda004z10R_c, dat42.lambda004z10M_removed, '--', color='k', label=r'$\lambda = 0.04$')
        cx[2].plot(dat[8].lambda008z10R_c, dat82.lambda008z10M_removed, '-.', color='k', label=r'$\lambda = 0.08$')
        cx[2].set_title('z = ' + '1.0', fontweight='bold')

        cx[2].legend(loc=1)

        for i in iter(cx):
            plot_styling(i, 'M_removed')

    if (flag == 'r_strip'):

        cx[0].plot(dat.theta00M10e13R_c, dat20.lambda002z00r_strip_r_dg, '-', color='k')

        cx[0].set_title('z = ' + '0.0', fontweight='bold')

        cx[1].plot(dat[3].lambda002z05R_c, dat21.lambda002z05r_strip_r_dg, '-', color='k')
        cx[1].plot(dat[4].lambda004z05R_c, dat41.lambda004z05r_strip_r_dg, '--', color='k')
        cx[1].plot(dat[5].lambda008z05R_c, dat81.lambda008z05r_strip_r_dg, '-.', color='k')
        cx[1].set_title('z = ' + '0.5', fontweight='bold')

        cx[2].plot(dat[6].lambda002z10R_c, dat22.lambda002z10r_strip_r_dg, '-', color='k', label=r'$\lambda = 0.02$')
        cx[2].plot(dat[7].lambda004z10R_c, dat42.lambda004z10r_strip_r_dg, '--', color='k', label=r'$\lambda = 0.04$')
        cx[2].plot(dat[8].lambda008z10R_c, dat82.lambda008z10r_strip_r_dg, '-.', color='k', label=r'$\lambda = 0.08$')
        cx[2].set_title('z = ' + '1.0', fontweight='bold')

        cx[2].legend(loc=2)

        for i in iter(cx):
            plot_styling(i, 'rstrip')


def Mass_frac(full_path1,posr):
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

    full_path1 = full_path1.replace('.dat','params.ini')

    parser = SafeConfigParser()
    if(full_path1 != './params.ini'):

                parser.read(full_path1)

                para = {}

                #print '\nCosmology:'
                for name in parser.options('cosmology'):
                    string_value = parser.get('cosmology', name)
                    value = parser.getfloat('cosmology', name)
                    para[name] = value
                    #print '  %-12s : %-7r -> %0.2f' % (name, string_value, value)

                #print '\nCluster'
                for name in parser.options('cluster'):
                    string_value = parser.get('cluster', name)
                    value = parser.getfloat('cluster', name)
                    para[name] = value
                    #print '  %-12s : %-7r -> %0.2f' % (name, string_value, value)

                #print '\ngalaxy'
                for name in parser.options('galaxy'):
                    string_value = parser.get('galaxy', name)
                    value = parser.getfloat('galaxy', name)
                    para[name] = value
                    #print '  %-12s : %-7r -> %0.2f' % (name, string_value, value)

                #print '\nmove'
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

                #print "n1,n = ", para['n1'], para['n']

                gal1 = Galaxy.Galaxy(para)




                m_d1 = gal1._f_g * gal1._m_d
                ymax = 20.0  # r/rd
                const = integrate.romberg(gal1.funct_sig_g, 0.0, ymax, divmax=100)
                sigg0 = (m_d1 / (2.0 * np.pi * (gal1._r_dg ** 2) * const)).value

                m1_total = (2*np.pi*sigg0*(gal1._r_dg.value**2))*integrate.romberg(gal1.funct_sig_g,posr,ymax,divmax=100)

                M_frac = m1_total

                return M_frac/(gal1._f_g*gal1._m_d.value)


def differentM(list1,En):

        mass =[]

        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.set_ylabel(r"$\mathbf{r_{strip}/r_{dg}}$", fontweight='bold', fontsize=16)
        ax1.set_xlabel(r"$\mathbf{R/R_{200}}$", fontweight='bold', fontsize=16)

        ax1.minorticks_on()
        ax1.tick_params(axis='both', which='minor', length=5, width=2, labelsize=15)
        ax1.tick_params(axis='both', which='major', length=8, width=2, labelsize=15)

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.set_ylabel(r"$\mathbf{M_{removed}}$", fontweight='bold', fontsize=16)
        ax2.set_xlabel(r"$\mathbf{R/R_{200}}$", fontweight='bold', fontsize=16)

        ax2.minorticks_on()
        ax2.tick_params(axis='both', which='minor', length=5, width=2, labelsize=15)
        ax2.tick_params(axis='both', which='major', length=8, width=2, labelsize=15)

        l = 0
        d={}
        for dirpath, dirnames, filenames in os.walk("."):
            for filename in [f for f in filenames if f.endswith(".dat")]:

                full_path = os.path.join(dirpath, filename)

                f = full_path.split('/')

                if(list1 == f[1:-1]):


                    print full_path

                    dat1 = np.genfromtxt(full_path)
                    # if(V in full_path):
                    u, indices = np.unique(dat1[:, 0], return_index=True)



                    dat1 = np.genfromtxt(full_path)
                    # if(V in full_path):
                    u, indices = np.unique(dat1[:, 0], return_index=True)


                    if(min(indices) == 0 and max(indices)!=0 ):

                        print full_path
                        ind = np.sort(indices)
                        M_removed = []

                        data = np.zeros((len(ind) + 1, 5), dtype=np.float)
                        j = 0



                        for i in range(1, len(ind)):
                            data[j, :] = dat1[ind[i] - 1, :]
                            M_removed.append(Mass_frac(full_path, data[j, 1]))
                            #print j
                            j = j + 1

                        data = data[data[:, 1] != 0.0]

                        #print M_removed,data


                        print full_path.split('_')[1].replace('.','')

                        d['M'+full_path.split('_')[1].replace('.','')+'R_c'] = data[:, 0]
                        d['M'+full_path.split('_')[1].replace('.','')+'r_strip_r_dg'] = data[:, 1]
                        d['M'+full_path.split('_')[1].replace('.','')+'M_removed'] = M_removed

                        #ax1.plot(data[:, 0], data[:, 1], '.', label='Mass = ' +full_path.split('_')[1]+r'$M_{\odot}$')
                        #ax2.plot(data[:, 0], M_removed, '.', label='Mass = ' + full_path.split('_')[1]+r'$M_{\odot}$')



        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))

        ax1.plot(df.M10e9R_c, df.M10e9r_strip_r_dg,label='Mass = ' +r'$1 \times 10^{9}$'+r'$M_{\odot}$',linewidth=2.0)
        ax1.plot(df.M10e10R_c,df.M10e10r_strip_r_dg,label='Mass = ' +r'$1 \times 10^{10}$'+r'$M_{\odot}$',linewidth=2.0)
        ax1.plot(df.M10e11R_c,df.M10e11r_strip_r_dg,label='Mass = ' +r'$1 \times 10^{11}$'+r'$M_{\odot}$',linewidth=2.0)
        ax1.plot(df.M40e11R_c,df.M40e11r_strip_r_dg,label='Mass = ' +r'$4 \times 10^{11}$'+r'$M_{\odot}$',linewidth=2.0)
        ax1.plot(df.M80e11R_c,df.M80e11r_strip_r_dg,label='Mass = ' +r'$8 \times 10^{11}$'+r'$M_{\odot}$',linewidth=2.0)
        ax1.plot(df.M10e12R_c,df.M10e12r_strip_r_dg,label='Mass = ' +r'$1 \times 10^{12}$'+r'$M_{\odot}$',linewidth=2.0)
        ax1.plot(df.M20e12R_c,df.M20e12r_strip_r_dg,label='Mass = ' +r'$2 \times 10^{12}$'+r'$M_{\odot}$',linewidth=2.0)
        ax1.plot(df.M40e12R_c,df.M40e12r_strip_r_dg,label='Mass = ' +r'$4 \times 10^{12}$'+r'$M_{\odot}$',linewidth=2.0)
        ax1.plot(df.M60e12R_c,df.M60e12r_strip_r_dg,label='Mass = ' +r'$6 \times 10^{12}$'+r'$M_{\odot}$',linewidth=2.0)
        ax1.plot(df.M80e12R_c,df.M80e12r_strip_r_dg,label='Mass = ' +r'$8 \times 10^{12}$'+r'$M_{\odot}$',linewidth=2.0)
        ax1.plot(df.M10e13R_c,df.M10e13r_strip_r_dg,label='Mass = ' +r'$1 \times 10^{13}$'+r'$M_{\odot}$',linewidth=2.0)

        ax2.plot(df.M10e9R_c, df.M10e9M_removed, label='Mass = ' + r'$1 \times 10^{9}$' + r'$M_{\odot}$',linewidth=2.0)
        ax2.plot(df.M10e10R_c, df.M10e10M_removed, label='Mass = ' + r'$1 \times 10^{10}$' + r'$M_{\odot}$',linewidth=2.0)
        ax2.plot(df.M10e11R_c, df.M10e11M_removed, label='Mass = ' + r'$1 \times 10^{11}$' + r'$M_{\odot}$',linewidth=2.0)
        ax2.plot(df.M40e11R_c, df.M40e11M_removed, label='Mass = ' + r'$4 \times 10^{11}$' + r'$M_{\odot}$',linewidth=2.0)
        ax2.plot(df.M80e11R_c, df.M80e11M_removed, label='Mass = ' + r'$8 \times 10^{11}$' + r'$M_{\odot}$',linewidth=2.0)
        ax2.plot(df.M10e12R_c, df.M10e12M_removed, label='Mass = ' + r'$1 \times 10^{12}$' + r'$M_{\odot}$',linewidth=2.0)
        ax2.plot(df.M20e12R_c, df.M20e12M_removed, label='Mass = ' + r'$2 \times 10^{12}$' + r'$M_{\odot}$',linewidth=2.0)
        ax2.plot(df.M40e12R_c, df.M40e12M_removed, label='Mass = ' + r'$4 \times 10^{12}$' + r'$M_{\odot}$',linewidth=2.0)
        ax2.plot(df.M60e12R_c, df.M60e12M_removed, label='Mass = ' + r'$6 \times 10^{12}$' + r'$M_{\odot}$',linewidth=2.0)
        ax2.plot(df.M80e12R_c, df.M80e12M_removed, label='Mass = ' + r'$8 \times 10^{12}$' + r'$M_{\odot}$',linewidth=2.0)
        ax2.plot(df.M10e13R_c, df.M10e13M_removed, label='Mass = ' + r'$1 \times 10^{13}$' + r'$M_{\odot}$',linewidth=2.0)

        ax1.legend(loc=2)

        ax2.legend(loc=3)

        fig1.tight_layout()
        fig2.tight_layout()

        fig1.savefig('Mvariation_R_vs_r_strip.eps',format = 'eps', dpi=600)
        fig2.savefig('Mvariation_R_vs_M_removed.eps',format='eps', dpi=600)

        fig1.savefig('Mvariation_R_vs_r_strip.png', dpi=600)
        fig2.savefig('Mvariation_R_vs_M_removed.png', dpi=600)

        plt.show()



def differentz(list2,M,V,En):

        fp2 = open('tmp2.txt', 'w')

        mass_frac = []

        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.set_ylabel(r"$\mathbf{r_{strip}/r_{dg}}$", fontweight='bold', fontsize=16)
        ax1.set_xlabel(r"$\mathbf{R/R_{200}}$", fontweight='bold', fontsize=16)

        ax1.minorticks_on()
        ax1.tick_params(axis='both', which='minor', length=5, width=2, labelsize=15)
        ax1.tick_params(axis='both', which='major', length=8, width=2, labelsize=15)

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.set_ylabel(r"$\mathbf{M_{removed}}$", fontweight='bold', fontsize=16)
        ax2.set_xlabel(r"$\mathbf{R/R_{200}}$", fontweight='bold', fontsize=16)

        ax2.minorticks_on()
        ax2.tick_params(axis='both', which='minor', length=5, width=2, labelsize=15)
        ax2.tick_params(axis='both', which='major', length=8, width=2, labelsize=15)

        l = 0
        z = []

        d = {}

        for dirpath, dirnames, filenames in os.walk("."):

            for filename in [f for f in filenames if f.endswith(".dat")]:


                full_path = os.path.join(dirpath, filename)

                f = full_path.split('/')


                if (list2 == f[1:-2]):

                    lines = ['k-', 'k--', 'k-.']
                    # print f[1:-2]
                    g = 0

                    if(M in full_path):

                        print full_path


                        z.append(f[5])

                        dat1 = np.genfromtxt(full_path)
                        #if(V in full_path):
                        u, indices = np.unique(dat1[:, 0], return_index=True)

                        ind = np.sort(indices)

                        #print indices
                        M_removed=[]

                        data = np.zeros((len(ind) + 1, 5), dtype=np.float)
                        j = 0
                        for i in range(1, len(ind)):
                            data[j, :] = dat1[ind[i] - 1, :]
                            M_removed.append(Mass_frac(full_path, data[j, 1]))
                            #print j
                            j = j + 1

                        #print M_removed,data[:,1]
                        data=data[data[:,1]!=0.0]

                        #print data.shape,len(M_removed)
                        d[f[5].replace('.','')+'R_c']= data[:,0]
                        d[f[5].replace('.','') + 'r_strip'] = data[:, 1]
                        d[f[5].replace('.','') + 'M_removed'] = M_removed

                        #ax1.plot(data[:,0],data[:,1],lines[g],linewidth=2.0,label='z = '+str(z[l][1:]))
                        #ax2.plot(data[:,0],M_removed,lines[g],linewidth=2.0,label='z = '+str(z[l][1:]))
                        #l=l+1
                        #g=g+1


        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))

        print list(df.columns.values)

        ax1.plot(df.z00R_c,df.z00r_strip,'k-',linewidth=2.0,label='z = 0.0')
        ax1.plot(df.z05R_c, df.z05r_strip, 'k--', linewidth=2.0, label='z = 0.5' )
        ax1.plot(df.z10R_c, df.z10r_strip, 'k-.', linewidth=2.0, label='z = 1.0')

        ax2.plot(df.z00R_c,df.z00M_removed,'k-',linewidth=2.0,label='z = 0.0' )
        ax2.plot(df.z05R_c, df.z05M_removed, 'k--', linewidth=2.0, label='z = 0.5')
        ax2.plot(df.z10R_c, df.z10M_removed, 'k-.', linewidth=2.0, label='z = 1.0')

        ax1.legend(loc=2)
        ax2.legend(loc=1)
        ax1.grid(True)
        ax2.grid(True)
        fig1.tight_layout()
        fig2.tight_layout()

        #plt.savefig('Zvariation'+ En + '_' + 'R_vs_rstrip_r_dg' + str(M) + '.png', dpi=600)
        fig1.savefig('Zvariation_R_vs_r_strip.eps',format='eps',dpi=600)
        fig2.savefig('Zvariation_R_vs_M_removed.eps', format='eps',dpi=600)

        fig1.savefig('Zvariation_R_vs_r_strip.png', dpi=600)
        fig2.savefig('Zvariation_R_vs_M_removed.png', dpi=600)

        plt.show()


def different_vel(list1):
    mass = []
    mass_frac = []
    d={}

    for dirpath, dirnames, filenames in os.walk("."):
        for filename in [f for f in filenames if f.endswith(".dat")]:
            full_path = os.path.join(dirpath, filename)

            f = full_path.split('/')

            #print f[1:-1],list1[:-1]

            #print f[3:-1],list1[1:-1]
            if(f[3:-2]==list1[1:-1]):

                if(f[1] == list1[0]):

                    if(full_path.split('_')[1]== list1[len(list1)-1]):

                        print full_path



                        dat1 = np.genfromtxt(full_path)
                        u, indices = np.unique(dat1[:, 0], return_index=True)

                        ind = np.sort(indices)

                        if (min(indices) == 0 and max(indices) != 0):

                            print full_path
                            ind = np.sort(indices)
                            M_removed = []

                            data = np.zeros((len(ind) + 1, 5), dtype=np.float)
                            j = 0

                            for i in range(1, len(ind)):
                                data[j, :] = dat1[ind[i] - 1, :]
                                M_removed.append(Mass_frac(full_path, data[j, 1]))
                                # print j
                                j = j + 1

                            data = data[data[:, 1] != 0.0]

                            # print M_removed,data


                            print full_path.split('_')[1].replace('.', '')

                            d[f[2].replace('.', '')+full_path.split('/')[7].replace('.', '')+ 'R_c'] = data[:, 0]
                            d[f[2].replace('.', '')+full_path.split('/')[7].replace('.', '')+ 'r_strip_r_dg'] = data[:, 1]
                            d[f[2].replace('.', '')+full_path.split('/')[7].replace('.', '')+ 'M_removed'] = M_removed

                        # ax1.plot(data[:, 0], data[:, 1], '.', label='Mass = ' +full_path.split('_')[1]+r'$M_{\odot}$')
                        # ax2.plot(data[:, 0], M_removed, '.', label='Mass = ' + full_path.split('_')[1]+r'$M_{\odot}$')

    return d



def different_Mass(list1):
    mass = []
    mass_frac = []
    d = {}

    for dirpath, dirnames, filenames in os.walk("."):
        for filename in [f for f in filenames if f.endswith(".dat")]:
            full_path = os.path.join(dirpath, filename)

            f = full_path.split('/')

            #print f[1:-1],list1

            if (f[1:-2] == list1):

                print full_path




                dat1 = np.genfromtxt(full_path)
                u, indices = np.unique(dat1[:, 0], return_index=True)

                ind = np.sort(indices)

                if (min(indices) == 0 and max(indices) != 0):

                        print full_path
                        ind = np.sort(indices)
                        M_removed = []

                        data = np.zeros((len(ind) + 1, 5), dtype=np.float)
                        j = 0

                        for i in range(1, len(ind)):
                            data[j, :] = dat1[ind[i] - 1, :]
                            M_removed.append(Mass_frac(full_path, data[j, 1]))
                            # print j
                            j = j + 1

                        data = data[data[:, 1] != 0.0]

                        # print M_removed,data


                        print full_path.split('_')[1].replace('.', '')

                        d['M' + full_path.split('_')[1].replace('.', '') + full_path.split('/')[7].replace('.', '') + 'R_c'] = data[:, 0]
                        d['M' + full_path.split('_')[1].replace('.', '') + full_path.split('/')[7].replace('.', '') + 'r_strip_r_dg'] = data[:, 1]
                        d['M' + full_path.split('_')[1].replace('.', '') + full_path.split('/')[7].replace('.', '') + 'M_removed'] = M_removed

                        # ax1.plot(data[:, 0], data[:, 1], '.', label='Mass = ' +full_path.split('_')[1]+r'$M_{\odot}$')
                        # ax2.plot(data[:, 0], M_removed, '.', label='Mass = ' + full_path.split('_')[1]+r'$M_{\odot}$')

    return d


def different_theta(list1):
    mass = []
    mass_frac = []
    d={}

    for dirpath, dirnames, filenames in os.walk("."):
        for filename in [f for f in filenames if f.endswith(".dat")]:
            full_path = os.path.join(dirpath, filename)

            f = full_path.split('/')

            #print f[1:-1],list1[:-1]

            if(f[2:-2]==list1[:-1]):

                if(full_path.split('_')[1]== list1[len(list1)-1]):
                    print full_path

                    dat1 = np.genfromtxt(full_path)
                    u, indices = np.unique(dat1[:, 0], return_index=True)

                    ind = np.sort(indices)

                    if (min(indices) == 0 and max(indices) != 0):

                        print full_path
                        ind = np.sort(indices)
                        M_removed = []

                        data = np.zeros((len(ind) + 1, 5), dtype=np.float)
                        j = 0

                        for i in range(1, len(ind)):
                            data[j, :] = dat1[ind[i] - 1, :]
                            M_removed.append(Mass_frac(full_path, data[j, 1]))
                            # print j
                            j = j + 1

                        data = data[data[:, 1] != 0.0]

                        # print M_removed,data


                        print full_path.split('_')[1].replace('.', '')

                        d[f[1].replace('.', '')+'M' + full_path.split('_')[1].replace('.', '')+ full_path.split('/')[7].replace('.', '')+ 'R_c'] = data[:, 0]
                        d[f[1].replace('.', '')+'M' + full_path.split('_')[1].replace('.', '')+full_path.split('/')[7].replace('.', '')+ 'r_strip_r_dg'] = data[:, 1]
                        d[f[1].replace('.', '')+'M' + full_path.split('_')[1].replace('.', '')+full_path.split('/')[7].replace('.', '')+ 'M_removed'] = M_removed

                        # ax1.plot(data[:, 0], data[:, 1], '.', label='Mass = ' +full_path.split('_')[1]+r'$M_{\odot}$')
                        # ax2.plot(data[:, 0], M_removed, '.', label='Mass = ' + full_path.split('_')[1]+r'$M_{\odot}$')

    return d


if __name__ == '__main__':

        print " what kind of plot do you want?? \n"
        print " 1. For one z, one velocity different M \n"
        print " 2. For one mass, one velocity different z\n"
        print " 4. For contour of z on M-V plane\n"
        print " 5. For comparision plot for different lambda\n"
        print " 6. For comparision plot for different Environment(cluster and group for del=1.0)\n"
        print " 8. For one z, one mass ,one v with different fg\n"
        print " 9. For comparision plot for different delta\n"


        print " a. For one velocity different theta and 1 Mass"
        print " b. For one theta different velocity and 1 Mass,"
        print " c. For one theta different Mass and 1 velocity,"

        type = raw_input('Please enter a number (1,2,3,4,5,6,7,8,9,10): ')

        if (type == 'c'):

            theta = str(input('theta?'))
            theta = 'theta' + theta

            vel = str(input('velocity?'))
            vel = 'vel' + vel



            lmbda = str(input('lambda?'))
            lmbda = 'lambda' + lmbda

            fg = str(input('fg?'))
            fg = 'fg' + fg

            En = raw_input("Environment")

            delta = raw_input('delta?')
            delta = 'delta' + delta

            #Mass = raw_input('For what Mass(solar units)?')



            #for v in ['vel0.0','vel0.5','vel1.0']:

            list1 = [theta,vel,lmbda, fg, En, delta]
            #different_Mass(list1)

            d = different_Mass(list1)

            df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))

            print list(df.columns.values)




            fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))

            fig2, (bx1, bx2, bx3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))


            # print list(df.columns.values)

            i = 0

            ax = [ax1, ax2, ax3]
            bx = [bx1, bx2, bx3]

            titl = ['z = 0.0', 'z = 0.5', 'z = 1.0']
            for z in ['z00', 'z05', 'z10']:
                j=0
                M=['$M = 1 \cdot 10^{10} M_{\odot}$', '$M = 1 \cdot 10^{11} M_{\odot}$','$M = 4 \cdot 10^{11} M_{\odot}$',\
                   '$M = 8 \cdot 10^{11} M_{\odot}$','$M = 1 \cdot 10^{12} M_{\odot}$','$M = 2 \cdot 10^{12} M_{\odot}$',\
                   '$M = 4 \cdot 10^{12} M_{\odot}$','$M = 6 \cdot 10^{12} M_{\odot}$','$M = 8 \cdot 10^{12} M_{\odot}$','$M = 1 \cdot 10^{13} M_{\odot}$']
                for Mass in ['M10e10','M10e11','M40e11','M80e11','M10e12','M20e12','M40e12','M60e12','M80e12','M10e13']:




                    R = Mass  + z + 'R_c'
                    rs = Mass + z + 'r_strip_r_dg'
                    Mr = Mass + z + 'M_removed'
                    dR = df[df.notnull()][R]
                    dr = df[df.notnull()][rs]
                    dM = df[df.notnull()][Mr]

                    ax[i].plot(dR, dr, linewidth=2.5, label=M[j])
                    bx[i].plot(dR, dM, linewidth=2.5, label=M[j])
                    ax[i].set_title(titl[i], fontweight='bold', fontsize=16)
                    bx[i].set_title(titl[i], fontweight='bold', fontsize=16)

                    #ax[i].set_xlim(0.0,2.05)
                    #ax[i].set_ylim(0.0,)

                    #bx[i].set_xlim(0.0, 2.05)
                    #bx[i].set_ylim(0.0,1.005)

                    j=j+1

                i = i + 1


            for cx in iter(ax):
                plot_styling(cx,'rstrip')
                cx.grid(True)
                #cx.legend(loc=2)

            for cx in iter(bx):
                plot_styling(cx,'M_removed')
                cx.grid(True)
                #cx.legend(loc=3)

            fig1.legend(loc=7)
            fig2.legend(loc=7)

            fig1.tight_layout()
            fig1.subplots_adjust(right=0.87)

            fig2.tight_layout()
            fig2.subplots_adjust(right=0.87)

            fig1.savefig(En+'AllMass' +theta +'velocityVariation_R_vs_r_strip.png', dpi=600)
            fig2.savefig(En+'AllMass' + theta+'velocityVariation_Mvariation_R_vs_M_removed.png', dpi=600)

            fig1.savefig(En+'AllMass' +theta +'velocityVariation_R_vs_r_strip.eps', format='eps', dpi=600)
            fig2.savefig(En+'AllMass' +theta+'velocityVariation_Mvariation_R_vs_M_removed.eps', format='eps', dpi=600)

            plt.show()






        if (type == 'b'):

            theta = str(input('theta?'))
            theta = 'theta' + theta

            lmbda = str(input('lambda?'))
            lmbda = 'lambda' + lmbda

            fg = str(input('fg?'))
            fg = 'fg' + fg

            En = raw_input("Environment")

            delta = raw_input('delta?')
            delta = 'delta' + delta

            Mass = raw_input('For what Mass(solar units)?')



            #for v in ['vel0.0','vel0.5','vel1.0']:

            list1 = [theta,lmbda, fg, En, delta,Mass]
            #different_vel(list1)

            d = different_vel(list1)

            df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))

            print list(df.columns.values)




            fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))

            fig2, (bx1, bx2, bx3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))


            # print list(df.columns.values)

            i = 0

            ax = [ax1, ax2, ax3]
            bx = [bx1, bx2, bx3]

            titl = ['z = 0.0', 'z = 0.5', 'z = 1.0']
            for z in ['z00', 'z05', 'z10']:
                j=0
                vel=['$v_{in}/V_{200}$ = 0.0', '$v_{in}/V_{200}$ = 0.5','$v_{in}/V_{200}$ = 1.0']
                for v in ['vel00','vel05','vel10']:


                    R = v  + z + 'R_c'
                    rs = v + z + 'r_strip_r_dg'
                    Mr = v + z + 'M_removed'
                    dR = df[df.notnull()][R]
                    dr = df[df.notnull()][rs]
                    dM = df[df.notnull()][Mr]

                    ax[i].plot(dR, dr, linewidth=2.5, label=vel[j])
                    bx[i].plot(dR, dM, linewidth=2.5, label=vel[j])
                    ax[i].set_title(titl[i], fontweight='bold', fontsize=16)
                    bx[i].set_title(titl[i], fontweight='bold', fontsize=16)

                    ax[i].set_xlim(0.0,2.05)
                    ax[i].set_ylim(0.0,)

                    bx[i].set_xlim(0.0, 2.05)
                    bx[i].set_ylim(0.0,1.005)

                    j=j+1

                i = i + 1


            for cx in iter(ax):
                plot_styling(cx,'rstrip')
                cx.grid(True)
                cx.legend(loc=2)

            for cx in iter(bx):
                plot_styling(cx,'M_removed')
                cx.grid(True)
                cx.legend(loc=1)

            fig1.savefig(En +theta +'velocityVariation_R_vs_r_strip.png', dpi=600)
            fig2.savefig(En + theta+'velocityVariation_Mvariation_R_vs_M_removed.png', dpi=600)

            fig1.savefig(En +theta +'velocityVariation_R_vs_r_strip.eps', format='eps', dpi=600)
            fig2.savefig(En +theta+'velocityVariation_Mvariation_R_vs_M_removed.eps', format='eps', dpi=600)

            plt.show()




        if(type=='a'):
            vel = str(input('velocity?'))
            vel = 'vel' + vel

            lmbda = str(input('lambda?'))
            lmbda = 'lambda' + lmbda

            fg = str(input('fg?'))
            fg = 'fg' + fg

            En = raw_input("Environment")

            delta = raw_input('delta?')
            delta = 'delta' + delta

            Mass = raw_input('For what Mass(solar units)?')


            list1 = [vel,lmbda,fg,En,delta,Mass]

            fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))

            fig2, (bx1, bx2, bx3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))


            d = different_theta(list1)

            df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))

            #print list(df.columns.values)

            i=0

            ax=[ax1,ax2,ax3]
            bx=[bx1, bx2, bx3]

            #fig,cx1 = plt.subplots(1,1)
            titl=['z = 0.0','z = 0.5','z = 1.0']
            for z in ['z00','z05','z10']:

                for th in ['theta00','theta50','theta100','theta150','theta200','theta300']:


                    R  = th + 'M' + Mass.replace('.', '') + z + 'R_c'
                    rs = th + 'M' + Mass.replace('.', '') + z + 'r_strip_r_dg'
                    Mr = th + 'M' + Mass.replace('.', '') + z + 'M_removed'
                    dR = df[df.notnull()][R]
                    dr = df[df.notnull()][rs]
                    dM = df[df.notnull()][Mr]

                    if(th=='theta50'):
                        theta = th.replace('theta50', r'$\theta =$5.0 $^\circ$')
                    else:
                        theta = th.replace('theta',r'$\theta = $')




                    ax[i].plot(dR,dr,linewidth=2.5,label=theta.replace('00','0.0 $^\circ$ '))
                    bx[i].plot(dR,dM,linewidth=2.5,label=theta.replace('00','0.0 $^\circ$ '))
                    ax[i].set_title(titl[i],fontweight='bold', fontsize=16)
                    bx[i].set_title(titl[i],fontweight='bold', fontsize=16)

                i=i+1

            for cx in iter(ax):
                plot_styling(cx,'rstrip')
                cx.grid(True)
                cx.legend(loc=2)

            for cx in iter(bx):
                plot_styling(cx,'M_removed')
                cx.grid(True)
                cx.legend(loc=1)

            fig1.savefig(En +vel +'ThetaVariation_R_vs_r_strip.png', dpi=600)
            fig2.savefig(En + vel+'ThetaVariation_Mvariation_R_vs_M_removed.png', dpi=600)

            fig1.savefig(En +vel +'ThetaVariation_R_vs_r_strip.eps', format='eps', dpi=600)
            fig2.savefig(En + vel+'ThetaVariation_Mvariation_R_vs_M_removed.eps', format='eps', dpi=600)

            plt.show()


        if(type=='9'):
            lmbda = str(input('lambda?'))
            lmbda = 'lambda' + lmbda
           
            En = raw_input("Environment")
            
            #delta = 'delta' + delta

            fg = str(input('fg?'))
            fg = 'fg' + fg
            
            


            Mass = raw_input('For what Mass(solar units)?')

            fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))

            fig2, (ax4, ax5, ax6) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))

            ax = [ax1,ax2,ax3]
            bx = [ax4,ax5,ax6]

            col = ['r', 'b']
            sty = ['o-','o-.']

            j = 1
            k = 0


            for redshift in ['0.0','0.5','1.0']:

                redshift1 = 'z' + redshift


                line = ['-','--']
                colors = ['k','k']
                i=0

                for delt in ['0.1','0.3']:
                
                    delta = 'delta' + delt

                    string1 = [lmbda,fg,En,delta,redshift1]

                    compare_delta(string1, Mass)

                    data,M_removed = compare_delta(string1, Mass)

                    ax[k].plot(data[:, 0], data[:, 1], line[i],color=colors[i], linewidth=1.5, label=r'$\delta$ = ' + delt)

                    ax[k].set_ylabel(r"$\mathbf{r_{strip}/r_{dg}}$", fontweight='bold', fontsize=16)
                    ax[k].set_xlabel(r"$\mathbf{R/R_{200}}$", fontweight='bold', fontsize=16)
                    ax[k].set_title('z = ' + redshift, fontweight='bold')
                    ax[k].minorticks_on()
                    ax[k].tick_params(axis='both', which='minor', length=3, width=2, labelsize=15)
                    ax[k].tick_params(axis='both', which='major', length=5, width=2, labelsize=15)

                    bx[k].plot(data[:, 0], M_removed, line[i],color=colors[i], linewidth=1.5, label=r'$\delta$ = ' +delt)
                    bx[k].set_ylabel(r"$\mathbf{M_{removed}}$", fontweight='bold', fontsize=16)
                    bx[k].set_xlabel(r"$\mathbf{R/R_{200}}$", fontweight='bold', fontsize=16)
                    bx[k].set_title('z = ' + redshift, fontweight='bold')
                    bx[k].minorticks_on()
                    bx[k].tick_params(axis='both', which='minor', length=3, width=2, labelsize=15)
                    bx[k].tick_params(axis='both', which='major', length=5, width=2, labelsize=15)

                    i = i + 1

                j = j + 1
                k = k + 1

            ax[k - 1].legend(loc=2)
            bx[k - 1].legend(loc=3)

            fig1.savefig(En+'DeltaVariation_R_vs_r_strip.png', dpi=600)
            fig2.savefig(En+'DeltaVariation_Mvariation_R_vs_M_removed.png', dpi=600)

            fig1.savefig(En+'DeltaVariation_R_vs_r_strip.eps',format='eps', dpi=600)
            fig2.savefig(En+'DeltaVariation_Mvariation_R_vs_M_removed.eps',format='eps', dpi=600)

            plt.show()


        if(type=='8'):
            lmbda = str(input('lambda?'))
            lmbda = 'lambda' + lmbda
           
            En = raw_input("Environment")
            delta = raw_input('delta?')
            delta = 'delta' + delta

           
            
            #redshift = raw_input('redshift?')
            #redshift = 'z' + redshift
            
            Mass = raw_input('For what Mass(solar units)?')

            fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))

            fig2, (ax4, ax5, ax6) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))

            ax = [ax1, ax2, ax3]
            bx = [ax4, ax5, ax6]

            #fig1,(ax1) = plt.subplots(1, 3, sharex=True, figsize=(14, 6))


            col = ['r', 'b']

            M_removed=[]

            j = 1
            k = 0


            for redshift in ['0.0', '0.5', '1.0']:
                redshift1 = 'z' + redshift

                i = 0

                lines = ['k-','k--']


                for f_g in ['0.1','0.3']:
                
                    t=f_g
                    f_g = 'fg' + f_g


                    string1 = [lmbda,f_g,En,delta,redshift1]
                    data,M_removed = compare_fg(string1,Mass)

                    ax[k].plot(data[:, 0], data[:, 1],lines[i], linewidth=1.5, label=r'$f_{g}$ = '+t)

                    ax[k].set_ylabel(r"$\mathbf{r_{strip}/r_{dg}}$", fontweight='bold', fontsize=16)
                    ax[k].set_xlabel(r"$\mathbf{R/R_{200}}$", fontweight='bold', fontsize=16)
                    ax[k].set_title('z = ' + redshift, fontweight='bold')
                    ax[k].minorticks_on()
                    ax[k].tick_params(axis='both', which='minor', length=3, width=2, labelsize=15)
                    ax[k].tick_params(axis='both', which='major', length=5, width=2, labelsize=15)

                    bx[k].plot(data[:, 0], M_removed,lines[i], linewidth=1.5,label=r'$f_{g}$ = '+t)
                    bx[k].set_ylabel(r"$\mathbf{M_{removed}}$", fontweight='bold', fontsize=16)
                    bx[k].set_xlabel(r"$\mathbf{R/R_{200}}$", fontweight='bold', fontsize=16)
                    bx[k].set_title('z = ' + redshift, fontweight='bold')
                    bx[k].minorticks_on()
                    bx[k].tick_params(axis='both', which='minor', length=3, width=2, labelsize=15)
                    bx[k].tick_params(axis='both', which='major', length=5, width=2, labelsize=15)

                    i = i + 1

                j = j + 1
                k = k + 1

            ax[k - 1].legend(loc=2)
            bx[k - 1].legend(loc=3)

            fig1.savefig('FgVariation_R_vs_r_strip.png', dpi=600)
            fig2.savefig('FgVariation_Mvariation_R_vs_M_removed.png', dpi=600)

            fig1.savefig('FgVariation_R_vs_r_strip.eps',format='eps' ,dpi=600)
            fig2.savefig('FgVariation_Mvariation_R_vs_M_removed.eps',format='eps', dpi=600)

            plt.show()

            '''
                         ax[k].plot(np.log10(data[:, 4]), data[:, 1],col[i]+'o' ,linewidth=2, ms=10.0,label=r'$f_{g}$ = '+t)
                         #ax[k].set_ylim(-0.1,4.0 )
                         ax[k].set_xlim(5, 14)
                         ax[k].set_ylabel(r"$\mathbf{r_{strip}/r_{dg}}$", fontweight='bold', fontsize=16)
                         ax[k].set_xlabel(r"$\mathbf{log (Mass(M_{\odot}))}$", fontweight='bold', fontsize=16)
                         #ax[k].set_title('z = '+redshift,fontweight= 'bold')
                         ax[k].minorticks_on()
                         ax[k].tick_params(axis='both', which='minor', length=3, width=2, labelsize=15)
                         ax[k].tick_params(axis='both', which='major', length=10, width=2, labelsize=15)
                         
                         #k=k+1
                         i=i+1
                         
                 
            #plt.suptitle('z = '+redshift+r'$v_{f}$',fontweight= 'bold')
            
            plt.legend(loc=2)

            fig1.savefig('comparefg' + '_' + 'M_vs_rstrip'+Vel+ 'all' + redshift + '.png',
                         dpi=600, bbox_inches='tight')

            plt.show()
            '''

        if(type=='7'):
            lmbda = str(input('lambda?'))
            lmbda = 'lambda' + lmbda
            fg = str(input('fg?'))
            fg = 'fg' + fg
            En = raw_input("Environment")
            delta = raw_input('delta?')
            delta = 'delta' + delta

            Mass = raw_input('For what Mass(solar units)?')

            fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))

            ax = [ax1, ax2, ax3]

            col = ['r', 'b', 'g']

            j = 1
            k = 0
            i=0
            for redshift in ['0.0','0.5','1.0']:
                red = 'z' + redshift


                string3 = [lmbda, fg, En, delta]
                data = compareV(string3, Mass, red)




                ax[k].semilogy(data[:, 4], data[:, 2], col[i] + 'o', linewidth=2, ms=10.0,label=r"$r_{out}$")
                ax[k].semilogy(data[:, 4], data[:, 3], col[i] + 'o', linewidth=2, ms=10.0,label=r"$r_{strip}$")
                # ax1.set_ylim(-0.05, )
                ax[k].set_xlim(0.5, 3.5)
                ax[k].set_ylabel(r"$\mathbf{log r(kpc)}$", fontweight='bold', fontsize=16)
                ax[k].set_xlabel(r"$\mathbf{v_{f}}$", fontweight='bold', fontsize=16)

                ax[k].set_title('z = ' + redshift, fontweight='bold')

                ax[k].minorticks_on()
                ax[k].tick_params(axis='both', which='minor', length=5, width=2, labelsize=15)
                ax[k].tick_params(axis='both', which='major', length=10, width=2, labelsize=15)
                
                ax[k].grid(b=True, which='major', linestyle='--')

                i = i + 1
                k = k + 1

            ax[k-1].legend(loc=2,fontsize=10)
            fig1.savefig('compareV' + '_' + 'V_vs_rstrip and r_out' + Mass + 'all' + redshift + '.png',
                         dpi=600, bbox_inches='tight')

           
            plt.show()


        if (type=='6'):

            lmbda = str(input('lambda?'))
            lmbda = 'lambda' + lmbda
            fg = str(input('fg?'))
            fg = 'fg' + fg

            delta = raw_input('delta?')
            delta = 'delta' + delta
            Mass = raw_input('For what Mass(solar units)?')

            #redshift = raw_input('redshift?')

            fig1,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True,figsize = (14,6))

            fig2, (ax4, ax5, ax6) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))

            ax = [ax1,ax2,ax3]
            bx = [ax4,ax5,ax6]


            col = ['r', 'b', 'g']

            j=1
            k=0

            for redshift in ['0.0','0.5','1.0']:

                red = 'z' + redshift

                i = 0

                lines=['k-','k--']

                for En in ['cluster','group']:

                    string1 = [lmbda, fg, En, delta]
                    data,M_removed = compareEn(string1, delta,red,Mass)


                    #ax[k] = fig1.add_subplot(1, 3, j)

                    ax[k].plot(data[:, 0], data[:, 1], lines[i],linewidth=1.5, label=En)
                    ax[k].set_ylabel(r"$\mathbf{r_{strip}/r_{dg}}$", fontweight='bold', fontsize=16)
                    ax[k].set_xlabel(r"$\mathbf{R/R_{200}}$", fontweight='bold', fontsize=16)

                    ax[k].set_title('z = ' + redshift, fontweight='bold')

                    ax[k].minorticks_on()
                    ax[k].tick_params(axis='both', which='minor', length=3, width=2, labelsize=15)
                    ax[k].tick_params(axis='both', which='major', length=5, width=2, labelsize=15)

                    bx[k].plot(data[:, 0], M_removed, lines[i],linewidth=1.5, label=En)
                    bx[k].set_ylabel(r"$\mathbf{M_{removed}}$", fontweight='bold', fontsize=16)
                    bx[k].set_xlabel(r"$\mathbf{R/R_{200}}$", fontweight='bold', fontsize=16)
                    bx[k].set_title('z = ' + redshift, fontweight='bold')
                    bx[k].minorticks_on()
                    bx[k].tick_params(axis='both', which='minor', length=3, width=2, labelsize=15)
                    bx[k].tick_params(axis='both', which='major', length=5, width=2, labelsize=15)

                    i = i + 1
                    
                j=j+1
                k=k+1

            ax[k - 1].legend(loc=2)
            bx[k - 1].legend(loc=3)

            fig1.savefig('EnVariation_R_vs_r_strip.png', dpi=600,bbox_inches='tight')
            fig2.savefig('EnVariation_R_vs_M_removed.png', dpi=600, bbox_inches='tight')

            fig1.savefig('EnVariation_R_vs_r_strip.eps',format='eps', dpi=600, bbox_inches='tight')
            fig2.savefig('EnVariation_R_vs_M_removed.eps',format='eps',dpi=600, bbox_inches='tight')

            plt.show()




        if (type=='5'):

            fg = str(input('fg?'))
            fg = 'fg' + fg
            En = raw_input("Environment")
            delta = raw_input('delta?')
            delta = 'delta' + delta

            Mass = raw_input('For what Mass(solar units)?')

            fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))

            fig2,(ax4,ax5,ax6) = plt.subplots(1,3,sharey=True,figsize=(14,6))


            ax = [ax1, ax2, ax3]
            bx = [ax4,ax5,ax6]



            col = ['r','b','g']

            d={}

            j = 1
            k = 0
            for redshift in ['0.0','0.5','1.0']:
                red = 'z' + redshift
                i=0
                for lm in ['0.02','0.04','0.08']:

                    lmbda = 'lambda' + lm
                    string1 = [lmbda, fg, En, delta, red]

                    #comparelambda(string1, Mass)


                    lmbda = 'lambda' + lm
                    string1 = [lmbda, fg, En, delta, red]
                    #print red,lmbda
                    data1,M_removed = comparelambda(string1,Mass)

                    d[lmbda.replace('.','')+red.replace('.','')+'R_c'] = data1[:,0]
                    d[lmbda.replace('.','')+red.replace('.','') + 'r_strip_r_dg'] = data1[:, 1]
                    d[lmbda.replace('.','')+red.replace('.','')+'M_removed'] = M_removed



                    #lmbda1 = pd.DataFrame(data[0],columns=[red])



            df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))

            print list(df.columns.values)

            dat20 = df[df.lambda002z00R_c.notnull()]
            dat21 = df[df.lambda002z05R_c.notnull()]
            dat22 = df[df.lambda002z10R_c.notnull()]
            dat40 = df[df.lambda004z00R_c.notnull()]
            dat41 = df[df.lambda004z05R_c.notnull()]
            dat42 = df[df.lambda004z10R_c.notnull()]
            dat80 = df[df.lambda008z00R_c.notnull()]
            dat81 = df[df.lambda008z05R_c.notnull()]
            dat82 = df[df.lambda008z10R_c.notnull()]

            dat = [dat20,dat40,dat80,dat21,dat41,dat81,dat22,dat42,dat82]

            different_lambda_plotter(ax,dat,'M_removed')
            different_lambda_plotter(bx, dat,'r_strip')


            fig1.savefig('Different_lambda_R_c_M_removed.eps',format='eps',dpi=600)
            fig2.savefig('Different_lambda_R_c_r_strip.eps', format='eps', dpi=600)

            fig1.savefig('Different_lambda_R_c_M_removed.png', dpi=600)
            fig2.savefig('Different_lambda_R_c_r_strip.png', dpi=600)

            plt.show()



        if (type == '1'):
            lmbda = str(input('lambda?'))
            lmbda = 'lambda' + lmbda
            fg = str(input('fg?'))
            fg = 'fg' + fg
            En = raw_input("Environment")
            delta = raw_input('delta?')
            delta = 'delta' + delta

            redshift = raw_input('redshift?')
            redshift = 'z' + redshift
            string1 = [lmbda, fg, En, delta, redshift]
            print string1
            differentM(string1,En)


        if (type == '2'):
            lmbda = str(input('lambda?'))
            lmbda = 'lambda' + lmbda
            fg = str(input('fg?'))
            fg = 'fg' + fg
            En = raw_input("Environment")
            delta = raw_input('delta?')
            delta = 'delta' + delta

            Mass = raw_input('For what Mass(solar units)?')
            Vel = raw_input('For what velocity(in units of circular velocity)?')
            Vel = '_v_' + Vel + '_'
            string2 = [lmbda, fg, En, delta]
            differentz(string2,Mass,Vel,En)



        if (type=='4'):
            lmbda = str(input('lambda?'))
            lmbda = 'lambda' + lmbda
            fg = str(input('fg?'))
            fg = 'fg' + fg
            En = raw_input("Environment")
            delta = raw_input('delta?')
            delta = 'delta' + delta

            redshift = raw_input('redshift?')
            redshift = 'z' + redshift
            list4 = [lmbda,fg,En,delta,redshift]
            print list4
            make_contour(redshift,list4)
