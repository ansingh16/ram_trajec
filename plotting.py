import numpy as np
import matplotlib.pyplot as plt


mass = [1.0e6,1.0e7,1.0e8,1.0e9,1.0e10,1.0e11,5.0e11,1.0e12,5.0e12,1.0e13]

data = np.genfromtxt('output.txt',dtype=np.float64)

fig1 = plt.figure(1)
plt.semilogx(mass,data[:,0],'r.')
plt.ylabel(r"$r_{c}/R_{200}$")
plt.xlabel(r"$Mass(M_{\odot})$")
plt.savefig('Mass_vs_rc.png')


fig2 = plt.figure(2)
plt.semilogx(mass,data[:,1],'r.')
plt.ylabel(r"$r_{mid}/r_{dg}$")
plt.xlabel(r"$Mass(M_{\odot})$")
plt.savefig('Mass_vs_rmid.png')

fig3 = plt.figure(3)
plt.semilogx(mass,data[:,2],'r.',label = r"$r_{out}$")
plt.semilogx(mass,data[:,3],'b.',label = r"$r_{mid}$")
plt.ylabel(r"$r(kpc)$")
plt.xlabel(r"$Mass(M_{\odot})$")
plt.legend()
plt.savefig('Mass_vs_rout_r_mid.png')
plt.show()

