import numpy as np
from scipy import integrate
from astropy import units as u
import cosmology


class Galaxy:
    def __init__(self, para):
        from astropy.constants import G

        cos = cosmology.Cosmology(para)

        self._G = G.to('Mpc3/(Msun Myr2)')

        self._lmbda = para['lmbda']
        self._z = para['z']
        self._alpha = para['alpha']
        self._f_g = para['f_g']
        self._md_tot = para['md_tot'] * u.Msun
        self._z_d = para['z_d'] * u.Mpc
        self._m_bh = para['m_bh'] * u.Msun
        self._m_bul = para['m_bul'] * u.Msun
        self._m_tot = self._m_bh + self._m_bul
        self._epsilon = para['epsilon']
        self._eps = para['eps']
        self._f_uni = para['f_uni']
        self._z = para['z']
        self._omega_m = para['omega_m']
        self._omega_l = para['omega_l']
        self._omega_k = para['omega_k']
        self._h = para['h']

        self._omega_m_z = (self._omega_m * ((1.0 + self._z) ** 3)) / (
        self._omega_m * ((1.0 + self._z) ** 3) + self._omega_k * ((1.0 + self._z) ** 2) + self._omega_l)
        d = self._omega_m_z - 1.0
        self._delta_c = 18.0 * np.pi ** 2 + 82.0 * d - 39.0 * d ** 2

        # G_gal = G.to('pc3/Msun yr2')

        self._m_d = self._alpha * self._f_uni * self._md_tot

        #print 'Total mass is galaxy is', self._md_tot, "total mass in disk = ", self._alpha * self._f_uni * self._md_tot, "mass is gas= ", self._f_g * self._m_d, "Mass in stars: ", (1.0 - self._f_g) * self._m_d
        #self._r_dg = ((self._lmbda/np.sqrt(2.0))*0.784*((self._md_tot*self._h/(1.0e+8*u.Msun))**(1.0/3.0))*((self._omega_m_z*18.0*np.pi**2/(self._omega_m*self._delta_c))**(1.0/3.0))*(10.0/(self._h*(1.0+self._z)))*u.kpc).to('Mpc')

        H = (cos.H(1.0 / (1 + cos._z))) * (1 / u.Myr)

        self._r_dg = (self._lmbda / np.sqrt(2.0)) * ((self._md_tot * 10.0 * self._G * H) ** (1.0 / 3.0)) / (10.0 * H)

        self._r_out = 10.0 * self._r_dg

        self._r_ds = self._r_dg / 2.0  # 3.0*self._r_out/20.0
        '''
        print 'r_dg is = ', self._r_dg
        print 'r_out is= ', self._r_out
        print 'Omega_m_z is = ', self._omega_m_z
        print 'delta_c is= ', self._delta_c
        '''

    def funct_sig_g(self, y1):
        f1 = y1 * np.exp(-y1)
        return f1

    def funct_sig_s(self, y2):
        f2 = y2 * np.exp(-y2)
        return f2

    def sig_g(self, x):
        m_d1 = self._f_g * self._m_d
        ymax = 10.0  # r/rd
        const = integrate.romberg(self.funct_sig_g, 0.0, ymax, divmax=100)
        self._sigg0 = (m_d1 / (2.0 * np.pi * (self._r_dg ** 2) * const)).value  # m_d1.value/(2.0*np.pi*(self._r_dg.value**2)) #(m_d1/(2.0*np.pi*(self._r_dg**2)*const)).value
        Sig_g = (self._sigg0 * np.exp((-x / self._r_dg).value)) * (u.Msun / u.Mpc ** 2)
        return Sig_g

    def sig_s(self, x):
        m_d2 = (1.0 - self._f_g) * self._m_d
        ymax = 20.0
        const = integrate.romberg(self.funct_sig_s, 0.0, ymax, divmax=1000)
        self._sigs0 = (m_d2 / (2.0 * np.pi * (
        self._r_ds ** 2) * const)).value  # m_d2.value/(2.0*np.pi*(self._r_ds.value**2))#(m_d2/(2.0*np.pi*(self._r_ds**2)*const)).value
        Sig_s = (self._sigs0 * np.exp((-x / self._r_ds).value)) * (u.Msun / u.Mpc ** 2)
        return Sig_s

    def RES_F(self, x2):
        f_s = 2.0 * np.pi * (self._G) * self.sig_g(x2.value) * self.sig_s(x2.value)
        f_b = (self._G) * self._m_tot * self.sig_g(x2.value) / (x2 ** 2.0)
        Res_f = (f_s + self._epsilon * f_b)
        return Res_f

