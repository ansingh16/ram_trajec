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
                IGM1 = IGM.Cluster(para)

                r_c200 = IGM1._R_200

                r_box = 4.0 * r_c200

                #print "r_box", r_box
                v_c200 = IGM1._V200

                t_total = r_box / v_c200

                v = v_c200 * para['vel_factor']

                #print "total time taken to cross cluster: ", (r_box / v_c200).to('Myr')

                dt = t_total / para['n1']
                dr = gal1._r_out / para['n']

                # print "steps :",dt,dr
                n = para['n']
                n1 = para['n1']

                data = np.genfromtxt(full_path1.replace('params.ini', '.dat'))

                #print data.shape

                m1_total = 0.0

                m_d1 = gal1._f_g * gal1._m_d
                ymax = 20.0  # r/rd
                const = integrate.romberg(gal1.funct_sig_g, 0.0, ymax, divmax=100)
                sigg0 = (m_d1 / (2.0 * np.pi * (gal1._r_dg ** 2) * const)).value  # m_d1.value/(2.0*np.pi*(self._r_dg.value**2)) #(m_d1/(2.0*np.pi*(self._r_dg**2)*const)).value


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

        for dirpath, dirnames, filenames in os.walk("."):
            for filename in [f for f in filenames if f.endswith(".dat")]:

                full_path = os.path.join(dirpath, filename)

                f = full_path.split('/')

                if(list1 == f[1:-1]):



                    dat1 = np.genfromtxt(full_path)
                    # if(V in full_path):
                    u, indices = np.unique(dat1[:, 0], return_index=True)


                    if(min(indices) == 0 and max(indices)!=0 ):

                        print full_path
                        #print indices

                        ind = np.sort(indices)

                        #print indices
                        M_removed = []

                        data = np.zeros((len(ind) + 1, 4), dtype=np.float)
                        j = 0

                        #print data
                        #print ind

                        for i in range(1, len(ind)):
                            data[j, :] = dat1[ind[i] - 1, :]
                            M_removed.append(Mass_frac(full_path, data[j, 1]))
                            #print j
                            j = j + 1

                        data = data[data[:, 1] != 0.0]

                        #print M_removed,data


                        ax1.plot(data[:, 0], data[:, 1], '.', label='Mass = ' +full_path.split('_')[1]+r'$M_{\odot}$')
                        ax2.plot(data[:, 0], M_removed, '.', label='Mass = ' + full_path.split('_')[1]+r'$M_{\odot}$')


        ax1.legend(loc=2)
        ax2.legend(loc=3)

        fig1.tight_layout()
        fig2.tight_layout()

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

        for dirpath, dirnames, filenames in os.walk("."):

            for filename in [f for f in filenames if f.endswith(".dat")]:


                full_path = os.path.join(dirpath, filename)

                f = full_path.split('/')

                #print f[1:-2]
                if (list2 == f[1:-2]):

                    if(M in full_path):

                        print full_path


                        z.append(f[5])

                        dat1 = np.genfromtxt(full_path)
                        #if(V in full_path):
                        u, indices = np.unique(dat1[:, 0], return_index=True)

                        ind = np.sort(indices)

                        #print indices
                        M_removed=[]

                        data = np.zeros((len(ind) + 1, 4), dtype=np.float)
                        j = 0
                        for i in range(1, len(ind)):
                            data[j, :] = dat1[ind[i] - 1, :]
                            M_removed.append(Mass_frac(full_path, data[j, 1]))
                            print j
                            j = j + 1

                        #print M_removed,data[:,1]
                        data=data[data[:,1]!=0.0]

                        print data.shape,len(M_removed)
                        ax1.plot(data[:,0],data[:,1],'.',label='z = '+str(z[l][1:]))
                        ax2.plot(data[:,0],M_removed,'.',label='z = '+str(z[l][1:]))
                        l=l+1

        ax1.legend(loc=2)
        ax2.legend(loc=1)
        ax1.grid(True)
        ax2.grid(True)
        fig1.tight_layout()
        fig2.tight_layout()

        #plt.savefig('Zvariation'+ En + '_' + 'R_vs_rstrip_r_dg' + str(M) + '.png', dpi=600)
        fig1.savefig('Zvariation_R_vs_r_strip.eps',format='eps',dpi=800)
        fig2.savefig('Zvariation_R_vs_M_removed.eps', format='eps',dpi=800)

        plt.show()


def make_contour(Z1,list4):

    fp4  = open('tmp4.dat',"w")

    for dirpath, dirnames, filenames in os.walk("."):
        for filename in [f for f in filenames if f.endswith(".dat")]:
            full_path = os.path.join(dirpath, filename)

            f = full_path.split('/')

            #print f

            if (list4 == f[1:-1]):
                print full_path

                words = full_path.split('_')
                #print words
                if(Z1 in full_path):

                      line = subprocess.check_output(['tail', '-1', full_path])
                      print full_path
                      fp4.writelines(words[1] + " " + words[3]+ " "+ line.split(' ')[2]+' '+ line.split(' ')[3]+'\n')


    fp4.close()

    import matplotlib
    import matplotlib
    from scipy.interpolate import griddata



    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'

    x,y,z1,z2 = np.genfromtxt('tmp4.dat', unpack=True)


    x = np.log10(x)
    xi = np.linspace(x.min(), x.max(), x.shape[0])
    yi = np.linspace(y.min(), y.max(), y.shape[0])

    X, Y = np.meshgrid(x, y)

    z = z2 / z1

    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    
    
    manual_locations = [(11.2, 2.2),(11.3,1.8)]

    #plt.imshow(zi)

    fig = plt.figure(figsize=(8,6))
    cs = plt.contour(xi, yi, zi,colors = 'black',linestyles = 'dashed',manual=True)
    plt.clabel(cs, inline=0, fontsize=16,fontweight='bold',rightside_up=True)
    
    plt.xlim(11.0, 12.0)

    plt.xlabel(r'$\mathbf{log(M(M_{\odot}))}$', fontweight='bold',fontsize=16)
    plt.ylabel(r'$\mathbf{v_{f}}$', fontweight='bold',fontsize=16)
    plt.savefig(lmbda+ En+'_'+'contour'+str(Z1)+'.png',dpi=600,bbox_inches='tight')

    plt.show()

    #manual_locations = [(12.6, 2.5), (12.14, 1.601), (12.233, 1.314), (12.467, 1.342), (12.7, 1.31)]
    
    '''

    from numpy.random import random_sample
    import numpy.ma as ma

    J = random_sample(X.shape)
    mask = J > 0.7
    X = ma.masked_array(X, mask=mask)
    Y = ma.masked_array(Y, mask=mask)
    Z = ma.masked_array(Z, mask=mask)



    #manual_locations = [(11.59, 1.347), (12.14, 1.601), (12.233, 1.314), (12.467, 1.342), (12.7, 1.31)]

    fig5 = plt.figure(5,figsize=(8,6))
    cs = plt.contour(X, Y, Z, cmap='ocean')
    plt.clabel(cs, inline=0, fontsize=20)# manual=manual_locations)
    plt.xlim(12.0, 13.0)
    plt.xlabel(r'$\mathbf{log(M(M_{\odot}))}$', fontweight='bold',fontsize=16)
    plt.ylabel(r'$\mathbf{V(v_{circ})}$', fontweight='bold',fontsize=16)

    #txt = r'countour:$r_{strip}$/$r_{out}$, $r_{strip}$ : radius of gas left after stripping,$r_{out}$ : outer radius of galaxy as it enters(10*disk scale radius), M: Mass of galaxy '+ 'z = ' + str(Z1.strip('z')) + ' V : velocity of galaxy(in circular velocity units)'

    #fig5.text(.5, 0.003, txt, ha='center')
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=15)
    plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=15)
    '''
   
    # plt.imshow(Z)
    #plt.show()


def comparelambda(list1,Mass):
    mass = []
    mass_frac = []

    fp1 = open('tmp1.txt', 'w')
    for dirpath, dirnames, filenames in os.walk("."):
        for filename in [f for f in filenames if f.endswith(".dat")]:

            full_path = os.path.join(dirpath, filename)

            f = full_path.split('/')

            if (list1 == f[1:-1]):

                if( Mass == full_path.split('_')[1]):


                    print full_path
                    '''
                    dat1 = np.genfromtxt(full_path)
                    # if(V in full_path):
                    u, indices = np.unique(dat1[:, 0], return_index=True)

                    ind = np.sort(indices)

                    # print indices
                    M_removed = []

                    data = np.zeros((len(ind) + 1, 4), dtype=np.float)
                    j = 0
                    for i in range(1, len(ind)):
                        data[j, :] = dat1[ind[i] - 1, :]
                        M_removed.append(Mass_frac(full_path, data[j, 1]))
                        print j
                        j = j + 1

                    # print M_removed,data[:,1]
                    data = data[data[:, 1] != 0.0]


    return data,M_removed
    '''


def compareEn(string1,D,Z,Mass):

    fp1 = open('tmp7.txt', 'w')

    mass_frac=[]
    mass=[]

    for dirpath, dirnames, filenames in os.walk("."):
        for filename in [f for f in filenames if f.endswith(".dat")]:

            full_path = os.path.join(dirpath, filename)

            words = full_path.split('/')

            if(string1[0:3]==words[1:4]):
                if (D in full_path):
                    if (Z in full_path):

                        if (Mass == full_path.split('_')[1]):

                            print full_path

                            dat1 = np.genfromtxt(full_path)
                            # if(V in full_path):
                            u, indices = np.unique(dat1[:, 0], return_index=True)

                            ind = np.sort(indices)

                            # print indices
                            M_removed = []

                            data = np.zeros((len(ind) + 1, 4), dtype=np.float)
                            j = 0
                            for i in range(1, len(ind)):
                                data[j, :] = dat1[ind[i] - 1, :]
                                M_removed.append(Mass_frac(full_path, data[j, 1]))
                                print j
                                j = j + 1

                            # print M_removed,data[:,1]
                            data = data[data[:, 1] != 0.0]

    return data, M_removed



def compare_fg(list1,Mass):

    mass=[]
    mass_frac=[]

    fgval=[]

    fp1=open('tmp.dat','w')

    
    for dirpath, dirnames, filenames in os.walk("."):
        for filename in [f for f in filenames if f.endswith(".dat")]:
            
            full_path = os.path.join(dirpath, filename)
            
            f = full_path.split('/')

            #print full_path,list1

            #print f[1:-1],list1[0:-1]

            #print list1[0:-1],f[1:-1],list1

            if(list1==f[1:-1]):

                if (Mass == full_path.split('_')[1]):

                    #print full_path.split('_')
                
                    print full_path

                    dat1 = np.genfromtxt(full_path)
                    # if(V in full_path):
                    u, indices = np.unique(dat1[:, 0], return_index=True)

                    ind = np.sort(indices)

                    # print indices
                    M_removed = []

                    data = np.zeros((len(ind) + 1, 4), dtype=np.float)
                    j = 0
                    for i in range(1, len(ind)):
                        data[j, :] = dat1[ind[i] - 1, :]
                        M_removed.append(Mass_frac(full_path, data[j, 1]))
                        print j
                        j = j + 1

                    # print M_removed,data[:,1]
                    data = data[data[:, 1] != 0.0]

    return data, M_removed



def compare_delta(list1,Mass):

    fp1=open('tmp.dat','w')

    mass=[]
    mass_frac=[]

    for dirpath, dirnames, filenames in os.walk("."):
        for filename in [f for f in filenames if f.endswith(".dat")]:
            
            full_path = os.path.join(dirpath, filename)
            
            f = full_path.split('/')

            #print full_path,list1

            #print f[1:6],list1

            if(f[1:6]==list1):

                if( Mass == full_path.split('_')[1]):

                    print full_path

                    dat1 = np.genfromtxt(full_path)
                    # if(V in full_path):
                    u, indices = np.unique(dat1[:, 0], return_index=True)

                    ind = np.sort(indices)

                    # print indices
                    M_removed = []

                    data = np.zeros((len(ind) + 1, 4), dtype=np.float)
                    j = 0
                    for i in range(1, len(ind)):
                        data[j, :] = dat1[ind[i] - 1, :]
                        M_removed.append(Mass_frac(full_path, data[j, 1]))
                        print j
                        j = j + 1

                    # print M_removed,data[:,1]
                    data = data[data[:, 1] != 0.0]

    return data, M_removed


def make_contour1(Z1,list4,bx=None):

                fp4  = open('tmp4.dat',"w")

                for dirpath, dirnames, filenames in os.walk("."):
                    for filename in [f for f in filenames if f.endswith(".dat")]:
                        full_path = os.path.join(dirpath, filename)
                        
                        f = full_path.split('/')
                        
                        #print f
                        
                        if (list4 == f[1:-1]):
                            #print full_path
                            
                            words = full_path.split('_')
                            
                            if(Z1 in full_path):
                                
                                line = subprocess.check_output(['tail', '-1', full_path])
                                #print full_path
                                #print words,line.split(' ')
                                fp4.writelines(words[1] + " " + words[3]+ " "+ line.split(' ')[2]+' '+ line.split(' ')[3]+'\n')


                fp4.close()

               
                from scipy.interpolate import griddata
                
                import matplotlib 

                matplotlib.rc('text', usetex=True)
                matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
                #plt.style.use('seaborn-talk')
                
                matplotlib.rc('text', usetex=True)
                matplotlib.rc('axes', linewidth=2)
                matplotlib.rc('font', weight='bold')

                matplotlib.rcParams['xtick.direction'] = 'out'
                matplotlib.rcParams['ytick.direction'] = 'out'
                
                x,y,z1,z2 = np.genfromtxt('tmp4.dat', unpack=True)
                
                
                x = np.log10(x)
                xi = np.linspace(x.min(), x.max(), x.shape[0])
                yi = np.linspace(y.min(), y.max(), y.shape[0])
                
                X, Y = np.meshgrid(x, y)
                
                z = z2 / z1
                
                zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
                
                
                if(list4[2]=='cluster'):
                    bx.set_xlim(11.6,13.0)
                    print "cluster"
                elif(list4[2]=='group'):
                    bx.set_xlim(11.0,12.0)
                    print "group"
                elif(list4[2]=='galaxy'):
                    bx.set_xlim(8.0,10.0)
                    print "galaxy"
                    
    
    
                #fig = plt.figure()
                cs = bx.contour(xi, yi, zi,colors = 'black',linestyles = 'dashed',manual=True)
                bx.clabel(cs,inline=0, fontsize=16,fontweight='bold',rightside_up=True)
                
                #bx.set_xlim(11.0, 12.0)
                
                bx.set_xlabel(r'$\mathbf{log(M(M_{\odot}))}$', fontweight='bold',fontsize=16)
                bx.set_ylabel(r'$\mathbf{v_{f}}$', fontweight='bold',fontsize=16)
                
                bx.set_title(list4[2],fontweight='bold',fontsize=16)


                bx.minorticks_on()
                bx.tick_params(axis='both', which='minor', length=3, width=2, labelsize=15)
                bx.tick_params(axis='both', which='major', length=5, width=2, labelsize=15)
                

                return cs
   



if __name__ == '__main__':

        print " what kind of plot do you want?? \n"
        print " 1. For one z, one velocity different M \n"
        print " 2. For one mass, one velocity different z\n"
        print " 4. For contour of z on M-V plane\n"
        print " 5. For comparision plot for different lambda\n"
        print " 6. For comparision plot for different Environment(cluster and group for del=1.0)\n"
        print " 8. For one z, one mass ,one v with different fg\n"
        print " 9. For comparision plot for different delta\n"
        print " 10. For comparision plot for different contour\n"

        type = raw_input('Please enter a number (1,2,3,4,5,6,7,8,9,10): ')


        if(type=='10'):
            
            lmbda = str(input('lambda?'))
            lmbda = 'lambda' + lmbda
            fg = str(input('fg?'))
            fg = 'fg' + fg
            #En = raw_input("Environment")
            delta = raw_input('delta?')
            delta = 'delta' + delta
            
            redshift = raw_input('redshift?')
            redshift = 'z' + redshift
            
            fig,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True,figsize=(14,6))
            
            ax = [ax3,ax2,ax1]
            
            
            
            i=2
            for En in ['cluster','group','galaxy']:
                
                if(En=='galaxy'):
                    delta='0.3'
                    delta = 'delta' + delta
                    
                list4 = [lmbda,fg,En,delta,redshift]
                    
                
                make_contour1(redshift,list4,ax[i])
                
                i=i-1

            plt.savefig('comparecontour'+lmbda+ '_' + 'contour'+ 'all' + redshift + '.png',dpi=600, bbox_inches='tight')
            
            plt.show(fig)




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
            i=0

            for redshift in ['0.0','0.5','1.0']:

                redshift1 = 'z' + redshift

                for delt in ['0.1','0.3']:
                
                    delta = 'delta' + delt

                    string1 = [lmbda,fg,En,delta,redshift1]

                    data,M_removed = compare_delta(string1, Mass)

                    ax[k].plot(data[:, 0], data[:, 1], '.', ms=8.0, label=r'$\delta$ = ' + delt)

                    ax[k].set_ylabel(r"$\mathbf{r_{strip}/r_{dg}}$", fontweight='bold', fontsize=16)
                    ax[k].set_xlabel(r"$\mathbf{R/R_{200}}$", fontweight='bold', fontsize=16)
                    ax[k].set_title('z = ' + redshift, fontweight='bold')
                    ax[k].minorticks_on()
                    ax[k].tick_params(axis='both', which='minor', length=3, width=2, labelsize=15)
                    ax[k].tick_params(axis='both', which='major', length=10, width=2, labelsize=15)

                    bx[k].plot(data[:, 0], M_removed, '.', ms=8.0, label=r'$\delta$ = ' +delt)
                    bx[k].set_ylabel(r"$\mathbf{M_{removed}}$", fontweight='bold', fontsize=16)
                    bx[k].set_xlabel(r"$\mathbf{R/R_{200}}$", fontweight='bold', fontsize=16)
                    bx[k].set_title('z = ' + redshift, fontweight='bold')
                    bx[k].minorticks_on()
                    bx[k].tick_params(axis='both', which='minor', length=3, width=2, labelsize=15)
                    bx[k].tick_params(axis='both', which='major', length=10, width=2, labelsize=15)

                    i = i + 1

                j = j + 1
                k = k + 1

            ax[k - 1].legend(loc=2)
            bx[k - 1].legend(loc=3)

            fig1.savefig('DeltaVariation_R_vs_r_strip.png', dpi=600)
            fig2.savefig('DeltaVariation_Mvariation_R_vs_M_removed.png', dpi=600)

            fig1.savefig('DeltaVariation_R_vs_r_strip.eps',format='eps', dpi=600)
            fig2.savefig('DeltaVariation_Mvariation_R_vs_M_removed.eps',format='eps', dpi=600)

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
            i=0

            for redshift in ['0.0', '0.5', '1.0']:
                redshift1 = 'z' + redshift

                for f_g in ['0.1','0.3']:
                
                    t=f_g
                    f_g = 'fg' + f_g


                    string1 = [lmbda,f_g,En,delta,redshift1]
                    data,M_removed = compare_fg(string1,Mass)

                    ax[k].plot(data[:, 0], data[:, 1],'.', ms=8.0, label=r'$f_{g}$ = '+t)

                    ax[k].set_ylabel(r"$\mathbf{r_{strip}/r_{dg}}$", fontweight='bold', fontsize=16)
                    ax[k].set_xlabel(r"$\mathbf{R/R_{200}}$", fontweight='bold', fontsize=16)
                    ax[k].set_title('z = ' + redshift, fontweight='bold')
                    ax[k].minorticks_on()
                    ax[k].tick_params(axis='both', which='minor', length=3, width=2, labelsize=15)
                    ax[k].tick_params(axis='both', which='major', length=10, width=2, labelsize=15)

                    bx[k].plot(data[:, 0], M_removed,'.', ms=8.0,label=r'$f_{g}$ = '+t)
                    bx[k].set_ylabel(r"$\mathbf{M_{removed}}$", fontweight='bold', fontsize=16)
                    bx[k].set_xlabel(r"$\mathbf{R/R_{200}}$", fontweight='bold', fontsize=16)
                    bx[k].set_title('z = ' + redshift, fontweight='bold')
                    bx[k].minorticks_on()
                    bx[k].tick_params(axis='both', which='minor', length=3, width=2, labelsize=15)
                    bx[k].tick_params(axis='both', which='major', length=10, width=2, labelsize=15)

                    i = i + 1

                j = j + 1
                k = k + 1

            ax[k - 1].legend(loc=2)
            bx[k - 1].legend(loc=3)

            fig1.savefig('FgVariation_R_vs_r_strip.png', dpi=600)
            fig2.savefig('FgVariation_Mvariation_R_vs_M_removed.png', dpi=600)

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
                for En in ['cluster','group']:

                    string1 = [lmbda, fg, En, delta]
                    data,M_removed = compareEn(string1, delta,red,Mass)


                    #ax[k] = fig1.add_subplot(1, 3, j)

                    ax[k].plot(data[:, 0], data[:, 1], col[i] + '.', ms=8.0, label=En)
                    ax[k].set_ylabel(r"$\mathbf{r_{strip}/r_{dg}}$", fontweight='bold', fontsize=16)
                    ax[k].set_xlabel(r"$\mathbf{R/R_{200}}$", fontweight='bold', fontsize=16)

                    ax[k].set_title('z = ' + redshift, fontweight='bold')

                    ax[k].minorticks_on()
                    ax[k].tick_params(axis='both', which='minor', length=3, width=2, labelsize=15)
                    ax[k].tick_params(axis='both', which='major', length=10, width=2, labelsize=15)

                    bx[k].plot(data[:, 0], M_removed, col[i] + '.', ms=8.0, label=En)
                    bx[k].set_ylabel(r"$\mathbf{M_{removed}}$", fontweight='bold', fontsize=16)
                    bx[k].set_xlabel(r"$\mathbf{R/R_{200}}$", fontweight='bold', fontsize=16)
                    bx[k].set_title('z = ' + redshift, fontweight='bold')
                    bx[k].minorticks_on()
                    bx[k].tick_params(axis='both', which='minor', length=3, width=2, labelsize=15)
                    bx[k].tick_params(axis='both', which='major', length=10, width=2, labelsize=15)

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

            j = 1
            k = 0
            for redshift in ['0.0', '0.5', '1.0']:
                red = 'z' + redshift
                i=0
                for lm in ['0.02','0.04','0.08']:

                    lmbda = 'lambda' + lm
                    string1 = [lmbda, fg, En, delta, red]

                    comparelambda(string1, Mass)

                    '''
                    lmbda = 'lambda' + lm
                    string1 = [lmbda, fg, En, delta, red]
                    data,M_removed = comparelambda(string1,Mass)



                    ax[k].plot(data[:, 0], data[:, 1],col[i]+'.', ms=8.0,label=lmbda)

                    ax[k].set_ylabel(r"$\mathbf{r_{strip}/r_{dg}}$", fontweight='bold', fontsize=16)
                    ax[k].set_xlabel(r"$\mathbf{R/R_{200}}$", fontweight='bold', fontsize=16)
                    ax[k].set_title('z = '+redshift,fontweight= 'bold')
                    ax[k].minorticks_on()
                    ax[k].tick_params(axis='both', which='minor', length=3, width=2, labelsize=15)
                    ax[k].tick_params(axis='both', which='major', length=10, width=2, labelsize=15)



                    bx[k].plot(data[:,0],M_removed,col[i]+'.' , ms=8.0,label=lmbda)
                    bx[k].set_ylabel(r"$\mathbf{M_{removed}}$", fontweight='bold', fontsize=16)
                    bx[k].set_xlabel(r"$\mathbf{R/R_{200}}$", fontweight='bold', fontsize=16)
                    bx[k].set_title('z = ' + redshift, fontweight='bold')
                    bx[k].minorticks_on()
                    bx[k].tick_params(axis='both', which='minor', length=3, width=2, labelsize=15)
                    bx[k].tick_params(axis='both', which='major', length=10, width=2, labelsize=15)

                    

                    i=i+1

                j = j + 1
                k = k + 1

            ax[k-1].legend(loc=2)
            bx[k-1].legend(loc=3)

            fig1.savefig('LambdaVariation_R_vs_r_strip.png', dpi=600)
            fig1.savefig('LambdaVariation_R_vs_r_strip.eps',format='eps', dpi=600)
            fig2.savefig('LambdaVariation_Mvariation_R_vs_M_removed.png', dpi=600)
            fig2.savefig('LambdaVariation_Mvariation_R_vs_M_removed.eps',format='eps', dpi=600)

            plt.show()
            '''

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
