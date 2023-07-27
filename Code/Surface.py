from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('surface.txt')


fig1,ax1 = plt.subplots(1,1,figsize=(10,6))


ax1.minorticks_on()
ax1.tick_params(axis='both', which='minor', length=3, width=2, labelsize=15)
ax1.tick_params(axis='both', which='major', length=10, width=2, labelsize=15)



ax1.plot(data[:,0],np.log10(data[:,1]),label=r'$\Sigma_{g}$')
ax1.plot(data[:,0],np.log10(data[:,2]),label=r'$\Sigma_{*}$')
ax1.legend()
ax1.set_xlabel(r"$\mathbf{r(Mpc)}$", fontweight='bold', fontsize=16)
ax1.set_ylabel(r"$\mathbf{log \Sigma (M_{\odot}/Mpc^{2})}$", fontweight='bold', fontsize=16)
plt.grid(True)
fig1.savefig('surface_density.png',dpi=600)




fig2,ax2 = plt.subplots(1,1,figsize=(10,6))
ax2.plot(data[:,0],np.log10(data[:,3]),label='Restoring Force')
ax2.set_xlabel(r"$\mathbf{r(Mpc)}$", fontweight='bold', fontsize=16)
ax2.set_ylabel(r"$\mathbf{log (F_{res}(\frac{M_{\odot}}{Mpc Myr^{2}}) = 2\pi\Sigma_{g}\Sigma_{*})}$)", fontweight='bold', fontsize=16)
plt.grid(True)
fig2.savefig('Res_force.png',dpi=600)



#print data[:,3],data[:,5]


u, indices = np.unique(data[:,4], return_index=True)

ind = np.sort(indices)

#print indices

data1 = np.zeros((len(ind)+1,3),dtype=np.float)
j=0
for i in range(1,len(ind)):
    data1[j,0] = data[ind[i]-1,4]
    data1[j,1] = data[ind[i]-1,3]
    data1[j,2] = data[ind[i]-1,5]
    
    j=j+1
    
data1 = data1[data1[:, 2] != 0.0]

#print data1

fig3,ax3 = plt.subplots(1,1,figsize=(10,6))
ax3.plot(data1[:,0],np.log10(data1[:,1]),label=r'$\mathbf{F_{res}}$')
ax3.plot(data1[:,0],np.log10(data1[:,2]),label=r'$\mathbf{F_{ram}}$')
ax3.set_xlabel(r"$\mathbf{R(Mpc)}$", fontweight='bold', fontsize=16)
ax3.set_ylabel(r"$\mathbf{log F(\frac{M_{\odot}}{Mpc Myr^{2}})}$", fontweight='bold', fontsize=16)
ax3.legend(loc=1)
plt.grid(True)

fig3.savefig('Res_force_vs_Ram_force.png',dpi=600)




plt.show()


