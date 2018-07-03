# -*- coding: utf-8 -*-
from __future__ import division, print_function
import copy
import numpy as np
import numpy as np
from munch import Munch
import h5py
from scipy import integrate, signal, linalg
from scipy.optimize import curve_fit
import sigutils
from sigutils._util import lin_or_logspace
from kpfm import lockin
from tqdm import tqdm
from ffta import pixel



def vary_dict(d, key, vals):
    """Generate a series of new dictionaries based on the dictionary 'd',
    with the 'key' taking each of the values in vals.

    Parameters
    ----------

    d: dictionary_like
        The base dictionary
    key: dictionary_key
        The key to vary
    vals: array_like
        The values which key will take on.

    Example
    -------

    >>> d = dict(a=2, b=3)
    >>> list(vary_dict(d, 'b', [1, 2, 3]))
    [{'a': 2, 'b': 1}, {'a': 2, 'b': 2}, {'a': 2, 'b': 3}]

    """
    for val in vals:
        new_dict = copy.copy(d)
        new_dict[key] = val
        yield new_dict


def scale_freq(phi, t, Hz=True, scale=1e6):
    """Calculate frequency shift from phase shift.

    Parameters
    ----------

    phi: array-like
        Phase array
    t: array-like
        Time array. Same size as phi.
    Hz: True, bool
        Return frequency in Hz if True. Frequency unit is rad/s if false.
        Default is True.

    """
    rad_per_s_to_Hz = 1/(2*np.pi) if Hz else 1
    return np.gradient(phi)/np.gradient(t) * scale * rad_per_s_to_Hz

def t_FP(trefm, t0=100):
    t = trefm.t[::trefm.dec]
    i = np.argmax(t > 100)
    m = slice(i, -1)
    dfdt = np.gradient(np.gradient(trefm.sim_phi_meas[m]*1e6)*1e6)
    return t[m][np.where(np.diff(np.sign(dfdt)))[0][0] - 1]


class DDHO(object):
    u"""
    
    Simulate cantilever dynamics for FFtrEFM. In the experiment, the light
    is turned on at t = 0 and the cantilever's resonance frequency and the DC
    force on the cantilever change as a result.
    
    The cantilever state is (x p)^T, the cantilever position and momemtum respectively.
    
    Parameters
    ----------
    
    omega_0: scalar 
        Cantilever angular frequency at t = 0 [rad/µs]
    k: scalar
        Cantilever spring constant [µN/µm]
    Q: scalar
        Cantilever quality factor
    omega_f: scalar
        Cantilever final angular frequency as t -> ∞ [rad/µs]
    omega_d: scalar
        Cantilever drive frequency [rad/µs]
    F_d: scalar
        Magnitude of the drive force [µN/µm]
    phi_d: scalar
        Drive phase, such that F(t) = F_d cos(omega_d * t + phi_d) [rad]
    tau: scalar
        Risetime of the frequency shift [µs]
    F_hv: scalar
        Magnitude of the force caused by turning on the light at t = 0.
    tau_F: scalar
        Risetime of the force caused by turning on the light at t = 0.
    """
    def __init__(self, omega_0, k, Q, omega_f, omega_d, F_d, phi_d, tau, F_hv, tau_F):
        self.omega_0 = omega_0
        self.k = k
        self.m = self.k / self.omega_0**2
        self.Q = Q
        self.omega_f = omega_f
        self.omega_d = omega_d
        self.F_d = F_d
        self.phi_d = phi_d
        self.tau = tau
        self.F_hv = F_hv
        self.tau_F = tau_F
        
    
    def __call__(self, x, t):
        if t <= 0:
            return np.array([
            x[1] / self.m,
            -self.omega_0**2 * self.m * x[0]
                - self.omega_0 / self.Q * x[1]
                + self.F_d * np.cos(self.omega_d*t+self.phi_d)
            ])
        else:
            return np.array([
                x[1]/self.m,
                -(self.omega_0 + (-(self.omega_f-self.omega_0)*np.expm1(-t/self.tau)))**2 * self.m * x[0] 
                    - self.omega_f / self.Q * x[1] 
                    + self.F_d * np.cos(self.omega_d*t+self.phi_d) - self.F_hv * np.expm1(-t/self.tau_F)
            ])

    def F_dc(self, t):
        return np.where(t >= 0, -self.F_hv * np.expm1(-t/self.tau_F), 0)
    
    def x_eq(self, t):
        return self.F_dc(t) / self.k
    
    def chi0(self, omega):
        return (1 - omega**2/self.omega_0**2 + 1j*omega / (self.omega_0 * self.Q))**-1
    
    def chi(self, omega):
        return (1 - omega**2/self.omega_f**2 + 1j*omega / (self.omega_0 * self.Q))**-1
    
    def phi(self, omega):
        return np.angle((1 - self.omega_d**2/omega**2 + 1j*self.omega_d / (omega * self.Q))**-1)
    
    def omega_t(self, t):
        return self.omega_0 + (self.omega_f - self.omega_0) * np.where(t > 0, 1-np.exp(-t/self.tau), 0)
    
    def z(self, x):
        return (x[:, 0] - 1j*x[:, 1]/(self.m * self.omega_d))
    
    def z_eq(self, x, t):
        return (x[:, 0] - self.x_eq(t) - 1j*x[:, 1]/(self.m * self.omega_d))
    
    def zLI(self, x, t):
        return self.z_eq(x, t) * np.exp(-1j*self.omega_d * t-self.phi_d*1j)
    
    def phase(self, x, t):
        return np.angle(self.zLI(x, t))
    
    def amp(self, x, t):
        return abs(self.zLI(x, t))
    
    def phit_func(self, t):
        return self.phi(self.omega_t(t))
    
    def phidot(self, x, t):
        return np.array([-0.5*self.omega_0/self.Q*(x[0] - self.phit_func(t))])
    
    def phi_approx(self, t):
        gamma = self.omega_0 / (2 * self.Q)
        tau = self.tau
        phi_0 = self.phit_func(0)
        phi_f = self.phit_func(max(1.0/gamma, tau)*10)
        gamma_tau = gamma*tau
        return np.where(t >= 0, 
                        phi_0 - (phi_f - phi_0) * 
                        (np.expm1(-gamma*t) - gamma_tau*np.expm1(-t/tau)) / (1 - gamma_tau),
                        phi_0
                       )
    
    def phi_approx_F(self, x, t):
        """This equation assumes tau = tau_F"""
        gamma = self.omega_0 / (2 * self.Q)
        tau = self.tau
        tau_F = self.tau_F
        phi_0 = self.phit_func(0)
        phi_f = self.phit_func(max(1.0/gamma, tau)*10)
        dx_hv = self.F_hv/self.k
        absolute_phi = phi_0 + self.phi_d
        
        ii = np.argmin(abs(t))
        z = self.z_eq(x[ii:ii+1], np.array([0]))[0]
        z_phi = np.angle(z)
        A = abs(z)
        dx = -dx_hv / (1 + (self.omega_0 * self.tau_F)**2)
        dy = dx_hv / (1 + (self.omega_0 * self.tau_F)**2) * self.omega_0 * self.tau_F
        
        gamma_tau = gamma*tau
        Delta_phi_F = -np.sin(z_phi) / A * dx + np.cos(z_phi) / A * dy
        self.Delta_phi_F = Delta_phi_F
        Delta_phi_diff = (phi_f - phi_0) - Delta_phi_F

        return np.where(t >=0, 
                phi_f - Delta_phi_F * np.exp(-t/tau) + Delta_phi_diff / (1 - gamma_tau) * (
                    gamma_tau * np.exp(-t/tau) - np.exp(-gamma*t))
                        ,phi_0)
        


class DDHOZ(object):
    u"""
    
    Simulate cantilever dynamics for a sample impedance measurement.
    

    Parameters
    ----------
    
    omega_0: scalar 
        Cantilever angular frequency at t = 0 [rad/µs]
    k: scalar
        Cantilever spring constant [µN/µm]
    Q: scalar
        Cantilever quality factor
    C: scalar
        Tip capacitance [pF]
    C2q: scalar
        Second derivative of the tip capacitance with respect to the vertical
        direct at constant charge [pF/µm²]
    C2D: scalar
        Second derivative of the tip capacitance with respect to the vertical
        direction at constant displacement [pF/µm²]
    Cs: scalar
        Sample capacitance [pF]
    Rs: scalar
        Sample resistance [MΩ]
    Vt: function
        Applied tip-sample voltage as a function of time [V]


    Function
    --------

    Call a created object with a state vector ``x`` and a time ``t``.
    The state vector has variables

        position [µm]
        momentum [ng.µm/µs]
        sample charge [pC]

    """
    def __init__(self, omega_0, k, Q, C, C2q, C2D, Cs, Rs, Vt):
        self.omega_0 = omega_0
        self.k = k
        self.m = self.k / self.omega_0**2
        self.Q = Q
        self.C = self.Ct = C
        self.C2q = C2q
        self.C2D = C2D
        self.Cz = -np.sqrt(0.5 * self.C * self.C2D)
        self.Cs = Cs
        self.Rs = Rs
        self.Vt = Vt
        self.omega_s = (self.Cs * self.Rs)**-1
        
    
    def __call__(self, x, t):
        q = self.tip_charge(x, t)
        return np.array([
            x[1] / self.m,
            -self.omega_0**2 * self.m * x[0]
                - self.omega_0 / self.Q * x[1]
                + 0.5 * self.C2q * q**2 / self.C**2 * x[0]
                + 0.5 * self.Cz * q**2 / self.C**2,
                self.omega_s * (q - x[2])
            ])

    def tip_charge(self, x, t):
        return ((x[2]/self.Cs + self.Vt(t))*
                (self.Ct**-1+self.Cs**-1- self.Ct**-2 * self.Cz * x[0])**-1
            )

    def tip_charge_0(self, t):
        """Tip charge to 0th order in displacement x"""
        return ((x[2]/self.Cs + self.Vt(t))*
                (self.Ct**-1+self.Cs**-1)**-1)

    def F_dc(self, x, t):
        q = np.array([self.tip_charge(xi, ti) for xi, ti in zip(x, t)])
        return 0.5 * self.Cz * q**2 / self.C**2
    
    def x_eq(self, x, t):
        return self.F_dc(x, t) / self.k
    
    def chi0(self, omega):
        return (1 - omega**2/self.omega_0**2 + 1j*omega / (self.omega_0 * self.Q))**-1
    
    # def chi(self, omega):
    #     return (1 - omega**2/self.omega_f**2 + 1j*omega / (self.omega_0 * self.Q))**-1
    
    def phi(self, omega):
        return np.angle((1 - self.omega_d**2/omega**2 + 1j*self.omega_d / (omega * self.Q))**-1)
    
    def omega_t(self, t):
        return self.omega_0 + (self.omega_f - self.omega_0) * np.where(t > 0, 1-np.exp(-t/self.tau), 0)
    
    def z(self, x):
        return (x[:, 0] - 1j*x[:, 1]/(self.m * self.omega_0))
    
    def z_eq(self, x, t):
        return (x[:, 0] - self.x_eq(x, t) - 1j*x[:, 1]/(self.m * self.omega_0))
    
    def zLI(self, x, t):
        return self.z_eq(x, t) * np.exp(-1j*self.omega_0 * t)
    
    def phase(self, x, t):
        return np.angle(self.zLI(x, t))
    
    def amp(self, x, t):
        return abs(self.zLI(x, t))
    
    def phit_func(self, t):
        return self.phi(self.omega_t(t))



class DDHOZs(object):
    u"""
    
    Simulate cantilever dynamics for a sample impedance measurement.
    

    Parameters
    ----------
    
    omega_0: scalar 
        Cantilever angular frequency at t = 0 [rad/µs]
    k: scalar
        Cantilever spring constant [µN/µm]
    Q: scalar
        Cantilever quality factor
    C: scalar
        Tip capacitance [pF]
    C2q: scalar
        Second derivative of the tip capacitance with respect to the vertical
        direct at constant charge [pF/µm²]
    C2D: scalar
        Second derivative of the tip capacitance with respect to the vertical
        direction at constant displacement [pF/µm²]
    Rs: scalar
        Sample resistance [MΩ]
    Vt: function
        Applied tip-sample voltage as a function of time [V]


    Function
    --------

    Call a created object with a state vector ``x`` and a time ``t``.
    The state vector has variables

        position [µm]
        momentum [ng.µm/µs]
        sample charge [pC]

    """
    def __init__(self, omega_0, k, Q, C, C2q, C2D, Rs, Vt):
        self.omega_0 = omega_0
        self.k = k
        self.m = self.k / self.omega_0**2
        self.Q = Q
        self.C = self.Ct = C
        self.C2q = C2q
        self.C2D = C2D
        self.Cz = -np.sqrt(0.5 * self.C * self.C2D)
        self.Rs = Rs
        self.Vt = Vt
        self.omega_q = (self.C * self.Rs)**-1
        
    
    def __call__(self, x, t):
        q = x[2]
        return np.array([
            x[1] / self.m,
            -self.omega_0**2 * self.m * x[0]
                - self.omega_0 / self.Q * x[1]
                + 0.5 * self.C2q * q**2 / self.C**2 * x[0]
                + 0.5 * self.Cz * q**2 / self.C**2,
                -self.omega_q * q + self.omega_q * q * self.Cz / self.C * x[0] + self.Vt(t)/self.Rs
            ])

    def tip_charge(self, x, t):
        return x[2]

    def tip_charge_0(self, t):
        """Tip charge to 0th order in displacement x"""
        return ((x[2]/self.Cs + self.Vt(t))*
                (self.Ct**-1+self.Cs**-1)**-1)

    def F_dc(self, x, t):
        q = np.array([self.tip_charge(xi, ti) for xi, ti in zip(x, t)])
        return 0.5 * self.Cz * q**2 / self.C**2
    
    def x_eq(self, x, t):
        return self.F_dc(x, t) / self.k
    
    def chi0(self, omega):
        return (1 - omega**2/self.omega_0**2 + 1j*omega / (self.omega_0 * self.Q))**-1
    
    # def chi(self, omega):
    #     return (1 - omega**2/self.omega_f**2 + 1j*omega / (self.omega_0 * self.Q))**-1
    
    def phi(self, omega):
        return np.angle((1 - self.omega_d**2/omega**2 + 1j*self.omega_d / (omega * self.Q))**-1)
    
    def omega_t(self, t):
        return self.omega_0 + (self.omega_f - self.omega_0) * np.where(t > 0, 1-np.exp(-t/self.tau), 0)
    
    def z(self, x):
        return (x[:, 0] - 1j*x[:, 1]/(self.m * self.omega_0))
    
    def z_eq(self, x, t):
        return (x[:, 0] - self.x_eq(x, t) - 1j*x[:, 1]/(self.m * self.omega_0))
    
    def zLI(self, x, t):
        return self.z_eq(x, t) * np.exp(-1j*self.omega_0 * t)
    
    def phase(self, x, t):
        return np.angle(self.zLI(x, t))
    
    def amp(self, x, t):
        return abs(self.zLI(x, t))
    
    def phit_func(self, t):
        return self.phi(self.omega_t(t))



def tf2rc(C, Cs, Rs, Ci, Ri, **kwargs):
    Ct = C
    b = np.array([Ci*Cs*Ri*Rs, Ci*Ri + Cs*(Ri + Rs), 1])
    a = np.array([Ci*(Cs + Ct)*Ri*Rs, Ci*Ri + (Cs + Ct)*(Ri + Rs), 1])
    return b, a


class DDHOZi(object):
    u"""
    
    Simulate cantilever dynamics for a sample impedance measurement.
    

    Parameters
    ----------
    
    omega_0: scalar 
        Cantilever angular frequency at t = 0 [rad/µs]
    k: scalar
        Cantilever spring constant [µN/µm]
    Q: scalar
        Cantilever quality factor
    C: scalar
        Tip capacitance [pF]
    C2q: scalar
        Second derivative of the tip capacitance with respect to the vertical
        direct at constant charge [pF/µm²]
    C2D: scalar
        Second derivative of the tip capacitance with respect to the vertical
        direction at constant displacement [pF/µm²]
    Cs: scalar
        Sample capacitance [pF]
    Rs: scalar
        Sample resistance [MΩ]
    Vt: function
        Applied tip-sample voltage as a function of time [V]
    Ci: scalar
        Sample ionic capacitance [pF]
    Ri: scalar
        Sample ionic resistance [MΩ]


    Function
    --------

    Call a created object with a state vector ``x`` and a time ``t``.
    The state vector has variables

        position [µm]
        momentum [ng.µm/µs]
        sample charge [pC]

    """
    def __init__(self, omega_0, k, Q, C, C2q, C2D, Cs, Rs, Vt, Ci, Ri):
        self.omega_0 = omega_0
        self.k = k
        self.m = self.k / self.omega_0**2
        self.Q = Q
        self.C = self.Ct = C
        self.C2q = C2q
        self.C2D = C2D
        self.Cz = -np.sqrt(0.5 * self.C * self.C2D)
        self.Cs = Cs
        self.Ci = Ci
        self.Rs = Rs
        self.Ri = Ri
        self.Vt = Vt
        self.omega_s = (self.Cs * self.Rs)**-1
    

    def tf(self):
        Ct = self.Ct
        Ci = self.Ci
        Cs = self.Cs
        Rs = self.Rs
        Ri = self.Ri
        return tf2rc(Ct, Cs, Rs, Ci, Ri)
    
    def __call__(self, x, t):
        q = self.tip_charge(x, t)
        Ct = self.Ct + x[0] * self.Cz
        Ci = self.Ci
        Cs = self.Cs
        Rs = self.Rs
        Ri = self.Ri
        qrs = x[2]
        qri = x[3]
        V = self.Vt(t)
        return np.array([
            x[1] / self.m,
            -self.omega_0**2 * self.m * x[0]
                - self.omega_0 / self.Q * x[1]
                + 0.5 * self.C2q * q**2 / self.C**2 * x[0]
                + 0.5 * self.Cz * q**2 / self.C**2,
                -(((-Cs - Ct)*qri)/(Ci*(Cs + Ct)*Rs)) - 
   ((Ci + Cs + Ct)*qrs)/(Ci*(Cs + Ct)*Rs) + (Ct*V)/((Cs + Ct)*Rs),
        -(qri/(Ci*Ri)) + qrs/(Ci*Ri)
            ])

    def tip_charge(self, x, t):
        qrs = x[2]
        qri = x[3]
        Ct = self.Ct + x[0] * self.Cz
        Cs = self.Cs
        V = self.Vt(t)
        return (Ct*qrs)/(Cs + Ct) + (Cs*Ct*V)/(Cs + Ct)

    def eq_charge(self, x=0, t=0):
        Ct = self.Ct + x * self.Cz
        V = self.Vt(t)
        return np.array([Ct * V, Ct * V])

    def eq_charge_freq(self, Vm, fm, phase, t=0):
        Ct = self.Ct
        Ci = self.Ci
        Cs = self.Cs
        Rs = self.Rs
        Ri = self.Ri
        s = 2j * np.pi * fm
        Vm = Vm * np.exp(1j*phase) * np.exp(-s * t)
        # Copied and pasted from Mathematica notebook 1707-lagrangian-2rc.nb
        charges = Vm * np.array([((Ct + Ci*Ct*Ri*s) / 
                (1 + (Cs + Ct)*(Ri + Rs)*s + Ci*Ri*s*(1 + (Cs + Ct)*Rs*s))
            ),
        Ct/(1 + (Cs + Ct)*(Ri + Rs)*s + Ci*Ri*s*(1 + (Cs + Ct)*Rs*s))
        ])
        # Value at t is the real part
        return charges.real 


    def eq_state(self, x=0, p=0, t=0):
        return np.r_[[x, p], self.eq_charge(x, t)]


    def eq_state_freq(self,  x=0, p=0, Vm=None, fm=None, phase=None, t=0):
        if Vm is None:
            Vm = self.Vm
        if fm is None:
            fm = self.fm
        if phase is None:
            phase = self.Vt_phase
        return np.r_[[x, p], self.eq_charge_freq(Vm, fm, phase, t)]

    # def tip_charge_0(self, t):
    #     """Tip charge to 0th order in displacement x"""
    #     return ((x[2]/self.Cs + self.Vt(t))*
    #             (self.Ct**-1+self.Cs**-1)**-1)

    def F_dc(self, x, t):
        q = np.array([self.tip_charge(xi, ti) for xi, ti in zip(x, t)])
        return 0.5 * self.Cz * q**2 / self.C**2
    
    def x_eq(self, x, t):
        return self.F_dc(x, t) / self.k
    
    def chi0(self, omega):
        return (1 - omega**2/self.omega_0**2 + 1j*omega / (self.omega_0 * self.Q))**-1
    
    # def chi(self, omega):
    #     return (1 - omega**2/self.omega_f**2 + 1j*omega / (self.omega_0 * self.Q))**-1
    
    def phi(self, omega):
        return np.angle((1 - self.omega_d**2/omega**2 + 1j*self.omega_d / (omega * self.Q))**-1)
    
    def omega_t(self, t):
        return self.omega_0 + (self.omega_f - self.omega_0) * np.where(t > 0, 1-np.exp(-t/self.tau), 0)
    
    def z(self, x):
        return (x[:, 0] - 1j*x[:, 1]/(self.m * self.omega_0))
    
    def z_eq(self, x, t):
        return (x[:, 0] - self.x_eq(x, t) - 1j*x[:, 1]/(self.m * self.omega_0))
    
    def zLI(self, x, t):
        return self.z_eq(x, t) * np.exp(-1j*self.omega_0 * t)
    
    def phase(self, x, t):
        return np.angle(self.zLI(x, t))
    
    def amp(self, x, t):
        return abs(self.zLI(x, t))
    
    def phit_func(self, t):
        return self.phi(self.omega_t(t))


def ringdown(t, A, k):
    return np.exp(-t*k*1e-6) * A

def makeV(V):
    def Vt(t):
        return V
    return Vt

class VoltageParabola(object):
    """
    To generate the voltage parabola, I need to 
    """
    def __init__(self, Vmin, Vmax, N, ddho_params, sim_params):
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.N = N
        self.Vts = np.linspace(self.Vmin, self.Vmax, self.N)
        self.Vts_func = [makeV(V) for V in self.Vts]
        self.ddho_params = ddho_params
        self.sim_params = sim_params
        self.ddhos = [DDHOZi(**params) for params in vary_dict(ddho_params, 'Vt', self.Vts_func)]


        self.out = Munch(freq=[], gamma=[], y0=[], y_out=[])
        for ddho in tqdm(self.ddhos):
            self.simulate(ddho, self.out)

        self.freq = self.fit_freq(self.Vts, self.out.freq)
        self.gamma = self.fit_gamma(self.Vts, self.out.gamma)


    def simulate(self, ddho, outputs):
        if 'y0' not in self.sim_params:
            if 'A' in self.sim_params:
                A = self.sim_params['A']
                y0 = ddho.eq_state(self.sim_params['A'])
            else:
                raise ValueError("Must specify cantilever amplitude or initial parameters")
        else:
            y0 = self.sim_params['y0']
            A = abs(y0[0] - 1j * y0[1]/ (ddho.omega_0 * ddho.m))


        t = self.sim_params['t']
        dt = np.mean(np.gradient(t))
        y_out = integrate.odeint(ddho, y0, t)


        popt, pcov = curve_fit(ringdown,
                           t,
                           abs(ddho.z_eq(y_out, t)),
                           p0=np.array([A, ddho.omega_0 / ddho.Q / 2 * 1e6]))

        gamma = popt[1]

        freq = np.mean(scale_freq(ddho.phase(y_out, t), t))

        outputs['gamma'].append(gamma)
        outputs['freq'].append(freq)
        outputs['y0'].append(y0)
        outputs['y_out'].append(y_out)

        return freq, gamma

    def fit_freq(self, V, freq):
        popt = np.polyfit(V, freq, 2)
        alpha = -popt[0]
        phi = popt[1] / (2*alpha)
        f0 = popt[2] + alpha * phi**2
        return Munch(alpha=alpha, phi=phi, f0=f0, p=popt)

    def fit_gamma(self, V, gamma):
        popt = np.polyfit(V, gamma, 2)
        alpha = popt[0]
        phi = -popt[1] / (2*alpha)
        f0 = popt[2] + alpha * phi**2
        return Munch(gamma_V=alpha, phi_A=phi, gamma0=f0, p=popt)


def makeVm(Vm, fm):
    def Vt(t):
        return Vm * np.sin(2*np.pi*fm*t)
    return Vt

def makeVam(Vm, fm, fam):
    def Vt(t):
        return Vm*(0.5 + 0.5*np.cos(2*np.pi*fam*t))*np.sin(2*np.pi*fm*t)
    return Vt

def lds_value(ba, fc, fm, C2q, C2D, Vm=1):
    H = lambda f: np.polyval(ba[0], 2j*np.pi*f) / np.polyval(ba[1], 2j*np.pi*f)

    return (C2q * Vm**2 / 32 * (H(fm)**2 + H(-fm)**2) + 
    C2D * Vm**2 / 64 * (H(fm) * H(fm - fc) + H(fm) * H(fm + fc)
                      + H(-fm)* H(-fc- fm) + H(-fm)* H(fc-fm)))

def lds_value2(ba, fc, fm, C2q, C2D, Vm=1):
    H = lambda f: np.polyval(ba[0], 2j*np.pi*f) / np.polyval(ba[1], 2j*np.pi*f)

    return (C2q * Vm**2 / 32 * (H(fm)**2 + H(-fm)**2) + 
    C2D * Vm**2 / 64 * (H(fm)**2 * H(fm - fc) + H(fm)**2 * H(fm + fc)
                      + H(-fm)**2 * H(-fc- fm) + H(-fm)**2 * H(fc-fm)))

class LDS(object):
    """
    To generate the voltage parabola, I need to 
    """
    def __init__(self, fmin, fmax, N, Vm, ddho_params, sim_params, pbar=None, full_output=False):
        self.fmin = fmin
        self.fmax = fmax
        self.N = N
        self.Vm = Vm
        self.fms = lin_or_logspace(self.fmin, self.fmax, self.N, log=True)
        self.Vts_func = [makeVm(Vm, fm) for fm in self.fms]

        self.ddho_params = ddho_params
        self.sim_params = sim_params
        self.full_output = full_output
        self.ddhos = []
        for fm, params in zip(self.fms, vary_dict(ddho_params, 'Vt', self.Vts_func)):
            ddho = DDHOZi(**params)
            ddho.fm = fm
            ddho.Vm = Vm
            ddho.Vt_phase = -np.pi/2
            self.ddhos.append(ddho)



        self.out = Munch(z=[], y0=[], y_out=[])
        if self.full_output:
            self.out['infodict']=[]
        for fm, ddho in zip(self.fms, self.ddhos):
            self.simulate(ddho, self.out, fm, full_output=self.full_output)
            if pbar is not None:
                pbar.update()

        for key, val in self.out.items():
            self.out[key] = np.array(val)

        # self.freq = self.fit_freq(self.Vts, self.out.freq)
        # self.gamma = self.fit_gamma(self.Vts, self.out.gamma)


    def simulate(self, ddho, outputs, fm, full_output=False):
        if 'y0' not in self.sim_params:
            if 'A' in self.sim_params:
                A = self.sim_params['A']
                if 'eq_state_freq' in self.sim_params:
                    y0 = ddho.eq_state_freq(x=A)
                else:
                    y0 = ddho.eq_state(x=A)
            else:
                raise ValueError("Must specify cantilever amplitude or initial parameters")
        else:
            y0 = self.sim_params['y0']
            A = abs(y0[0] - 1j * y0[1]/ (ddho.omega_0 * ddho.m))

        t = self.sim_params['t']
        dt = np.mean(np.gradient(t))
        if full_output:
            y_out, infodict = integrate.odeint(ddho, y0, t, full_output=True)
        else:
            y_out = integrate.odeint(ddho, y0, t)

        freq = scale_freq(ddho.phase(y_out, t), t)



        A = np.c_[np.cos(2*np.pi*fm*2*t), np.sin(2*np.pi*fm*2*t),
                    np.ones_like(t),
                    np.cos(2*np.pi*fm*t), np.sin(2*np.pi*fm*t),
                    np.cos(2*np.pi*3*fm*t), np.sin(2*np.pi*3*fm*t),
                    np.cos(2*np.pi*4*fm*t), np.sin(2*np.pi*4*fm*t),]
        out = linalg.lstsq(A, freq)
        z = np.dot(out[0], np.array([1, 1j, 0, 0, 0, 0, 0, 0, 0]))


        outputs['z'].append(z)
        outputs['y0'].append(y0)
        outputs['y_out'].append(y_out)
        if full_output:
            outputs['infodict'].append(infodict)



class BLDS(object):
    """
    """
    def __init__(self, fmin, fmax, N, Vm, fam, ddho_params, sim_params, pbar=None, full_output=False):
        self.fmin = fmin
        self.fmax = fmax
        self.N = N
        self.Vm = Vm
        self.fam = fam
        self.fms = lin_or_logspace(self.fmin, self.fmax, self.N, log=True)
        self.Vts_func = [makeVam(Vm, fm, fam) for fm in self.fms]

        self.ddho_params = ddho_params
        self.sim_params = sim_params
        self.full_output = full_output
        self.ddhos = []
        for fm, params in zip(self.fms, vary_dict(ddho_params, 'Vt', self.Vts_func)):
            ddho = DDHOZi(**params)
            ddho.fm = fm
            ddho.Vm = Vm
            ddho.Vt_phase = -np.pi/2
            self.ddhos.append(ddho)

        self.out = Munch(z=[], y0=[], y_out=[])
        if self.full_output:
            self.out['infodict']=[]
        for fm, ddho in zip(self.fms, self.ddhos):
            self.simulate(ddho, self.out, fam, full_output=self.full_output)
            if pbar is not None:
                pbar.update()

        for key, val in self.out.items():
            self.out[key] = np.array(val)

        # self.freq = self.fit_freq(self.Vts, self.out.freq)
        # self.gamma = self.fit_gamma(self.Vts, self.out.gamma)


    def simulate(self, ddho, outputs, fam, full_output=False):
        if 'y0' not in self.sim_params:
            if 'A' in self.sim_params:
                A = self.sim_params['A']
                if 'eq_state_freq' in self.sim_params:
                    y0 = ddho.eq_state_freq(x=A)
                else:
                    y0 = ddho.eq_state(x=A)
            else:
                raise ValueError("Must specify cantilever amplitude or initial parameters")
        else:
            y0 = self.sim_params['y0']
            A = abs(y0[0] - 1j * y0[1]/ (ddho.omega_0 * ddho.m))

        t = self.sim_params['t']
        dt = np.mean(np.gradient(t))
        if full_output:
            y_out, infodict = integrate.odeint(ddho, y0, t, full_output=True)
        else:
            y_out = integrate.odeint(ddho, y0, t)

        freq = scale_freq(ddho.phase(y_out, t), t)



        A = np.c_[np.cos(2*np.pi*fam*t), np.sin(2*np.pi*fam*t),
                    np.ones_like(t),
                    np.cos(2*np.pi*2*fam*t), np.sin(2*np.pi*2*fam*t),
                    np.cos(2*np.pi*3*fam*t), np.sin(2*np.pi*3*fam*t),
                    np.cos(2*np.pi*4*fam*t), np.sin(2*np.pi*4*fam*t),]
        out = linalg.lstsq(A, freq)
        z = np.dot(out[0], np.array([1, 1j, 0, 0, 0, 0, 0, 0, 0]))


        outputs['z'].append(z)
        outputs['y0'].append(y0)
        outputs['y_out'].append(y_out)
        if full_output:
            outputs['infodict'].append(infodict)



class TrEFM(object):
    def __init__(self, ddho, sim_params, workup_params, m=slice(2000, -2000)):
        self.ddho = ddho
        self.sim_params = sim_params
        self.x0 = sim_params['x0']
        self.t = sim_params['t']
        self.i0 = np.argmin(abs(self.t))
        self.workup_params = workup_params

        
        self.xv = integrate.odeint(self.ddho, self.x0, self.t)
        self.xv0 = self.xv[self.i0]
        self.m = m
        
        # Default to no decimation
        self.dec = self.sim_params.get("dec", 1)
        self.workup_params['sampling_rate'] = 1.0 / (self.sim_params['dt'] * self.dec)
        self.workup_params['total_time'] = self.sim_params['T']
        self.workup_params['trigger'] = -self.sim_params['t'][0]
        self.workup_params['drive_freq'] = self.ddho.omega_d/(2*np.pi)

        self.trefm_workup(self.dec)
        self.sim_phase = self.ddho.phase(self.xv, self.t)
        self.sim_phase_filt = signal.fftconvolve(self.sim_phase, self.sim_params['fir'], 
                                                 mode='same')
        self.phase_approx = self.ddho.phi_approx(self.t)
        self.phase_approx_F = self.ddho.phi_approx_F(self.xv, self.t)
        
        try:
            self.sim_phi_meas = signal.fftconvolve(
                self.sim_phase_filt[::self.dec], self.sim_params['fir_meas'], mode='full')[:-(self.sim_params['fir_meas'].size-1)]
            self.sim_phi_approx = signal.fftconvolve(
                self.phase_approx_F[::self.dec], self.sim_params['fir_meas'], mode='full')[:-(self.sim_params['fir_meas'].size-1)]
        except Exception as e:
            print(e.__doc__)
            print(e.message)
    
    def __call__(self, x):
        return getattr(self, x)[self.m]
        
    
    def trefm_workup(self, dec):
        p = pixel.Pixel(self.xv[:, 0].reshape(-1, 1)[::dec], self.workup_params)
        p.remove_dc()
        p.average()
        p.remove_dc()
        # p.check_drive_freq()
        p.apply_window()
        p.fir_filter()
        p.hilbert_transform()
        p.calculate_phase(correct_slope=True)
        p.calculate_inst_freq()
        self.p = p

        # Extract a copy of the bandpass filter used by the FFtrEFM workup
        # Code from from ffta.pixel.fir_filter
        nyq_rate = 0.5 * p.sampling_rate
        bw_half = p.filter_bandwidth / 2

        freq_low = (p.drive_freq - bw_half) / nyq_rate
        freq_high = (p.drive_freq + bw_half) / nyq_rate

        band = [freq_low, freq_high]
        self.trefm_fir = signal.firwin(p.n_taps, band, pass_zero=False,
                          window='parzen')

        self.phase = self.p.phase
        self.freq = self.p.inst_freq
        self.tout = self.t[::dec] + (self.workup_params['n_taps']-1)/2*self.sim_params['dt']*dec


class PkEFM(object):
    def __init__(self, ddho, sim_params, workup_params, m=slice(2000, -2000)):
        self.ddho = ddho
        self.sim_params = sim_params
        self.x0 = sim_params['x0']
        self.t = sim_params['t']
        self.i0 = np.argmin(abs(self.t))
        self.workup_params = workup_params

        
        self.xv = integrate.odeint(self.ddho, self.x0, self.t)
        self.xv0 = self.xv[self.i0]
        self.m = m
        
        # Default to no decimation
        self.dec = self.sim_params.get("dec", 1)

        self.trefm_workup(self.dec)
        self.sim_phase = self.ddho.phase(self.xv, self.t)
        self.sim_phase_filt = signal.fftconvolve(self.sim_phase, self.sim_params['fir'], 
                                                 mode='same')
        self.phase_approx = self.ddho.phi_approx(self.t)
        self.phase_approx_F = self.ddho.phi_approx_F(self.xv, self.t)
        
        try:
            self.sim_phi_meas = signal.fftconvolve(
                self.sim_phase_filt[::self.dec], self.sim_params['fir_meas'], mode='full')[:-(self.sim_params['fir_meas'].size-1)]
            self.sim_phi_approx = signal.fftconvolve(
                self.phase_approx_F[::self.dec], self.sim_params['fir_meas'], mode='full')[:-(self.sim_params['fir_meas'].size-1)]
        except Exception as e:
            print(e.__doc__)
            print(e.message)
    
    def __call__(self, x):
        return getattr(self, x)[self.m]


def fit_harmonics(z, wm, t):
    wm = float(wm)
    At = np.c_[np.ones_like(t), t,
              np.cos(wm*t), np.sin(wm*t),
              np.cos(wm*t*2), np.sin(wm*t*2),
              np.cos(wm*t*3), np.sin(wm*t*3),
              np.cos(wm*t*4), np.sin(wm*t*4),
              np.cos(wm*t*5), np.sin(wm*t*5)
             ]
    m = Munch(z=z, wm=wm, t=t, At=At)
    m.dphi = np.angle(z)
    m.dA = abs(z)
    m.fit_phi = linalg.lstsq(At, m.dphi)
    m.x_phi = m.fit_phi[0]
    m.fit_A = linalg.lstsq(At, m.dA)
    m.x_A = m.fit_A[0]
    m.df1, m.df2 = harmonics(m.x_phi) * wm / (2*np.pi) * 1e6
    m.dA1, m.dA2 = harmonics(m.x_A)
    return m


def ddho(subs_numerical, tt):
    wc = float(subs_numerical[omega_c])
    wm = float(subs_numerical[omega_m])
    ddhoz = DDHOZs(wc, float(subs_numerical[k]), Q=np.inf,
      C2q=6.5e-5, C2D=6.5e-5, C=float(subs_numerical[C]), Rs=float(1.0/(subs_numerical[C]*subs_numerical[omega_q])),
      Vt=lambda t: 5 * np.cos(wm*t)
             )
    y0 = np.array([np.cos(float(subs_numerical[phi_x])),
               float(
                -sm.sin(subs_numerical[phi_x]) * subs_numerical[m] * subs_numerical[A] * subs_numerical[omega_c]
                ),
               float(
                sm.re(C*V*H(omega_m)).subs({H:Hsub}).subs(subs_numerical)
                )
               ])

    yyt_numerical = integrate.odeint(ddhoz, y0, tt)
    z_num = (yyt_numerical[:, 0] - 1j * yyt_numerical[:, 1] / float(
    (m*omega_c).subs(subs_numerical)))

    z_num_ref = np.exp(-1j*np.pi/2) * np.exp(-1j*wc*tt)
    dz_num = z_num * z_num_ref
    m_num = fit_harmonics(dz_num, wm, tt)
    m_num.y = yyt_numerical
    m_num.t = tt
    m_num.wm = wm
    m_num.wc = wc
    return m_num