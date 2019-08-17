# coding: utf8
#from __future__ import absolute_import

import sympy as sp
import numpy as np
from scipy.integrate import odeint
import pylab as pl
import pickle
from IPython import embed as IPS

from control_aux import symb_tools as st
from control_aux.unicycle_controller_sim_prep import poly_matr_to_state_funcs


"""
Skript zur Simulation des Doppelpendels auf Rädern (nichtlineares Modell)
inkl. dynamischem Regler
"""

class SimModel(object):
    """
    Klasse, welche alle zum Modell gehöhrenden Daten (z.b. Paramerter-Werte)
    und Methoden (rhs_fnc) kapselt
    """

    def __init__(self):
        self.param_values = pdict_eqn['numparams']
        self.state_dim = 2*len(pdict_eqn['qs'])
        self.number_of_controller_states = None

    def create_simfunction(self,  Ak, Bk, symb):#*+
        """
        Erzeugt rechte Seite des Zustands-Systems
        (für Strecke und Beobachter)
        """

        n = self.state_dim

        # F0 kann nicht beeinflusst werden. -> ersten Einrtrag ignorieren
        input_symbs = list(pdict_eqn['extforce_list'])[1:]

        args = list(pdict_eqn['qs'])+list(pdict_eqn['qds'])+ input_symbs

        # F0 = 0 im Ausdruck setzen
        self.param_values.update({'F0': 0})

        qdd_expr = pdict_eqn['rhs'].subs(self.param_values)

        assert st.matrix_atoms(qdd_expr, sp.Symbol).issubset( set(args) )

        # Funktion, die die Beschleunigung in Abhängigkeit von q, qd, u
        # berechnet (nichtlineare Bewegungsgleichung)
        qdd_fnc = sp.lambdify(args, list(qdd_expr), modules="numpy")

        # Regler in Zustandsdarstellung bringen
        controller_rhs, orig_controller_output = \
                                    poly_matr_to_state_funcs(Ak, Bk, symb)
        self.number_of_controller_states = controller_rhs.state_dim#*-

        def rhs(state, time): #*+

            # Zustände der Strecke
            q = state[:n/2].T
            qd = state[n/2:n].T

            plant_state = np.concatenate([q, qd])

            # Differenz zum Sollzustand:

            # Soll-Zustand zur aktuellen Zeit:
            des_state = st.to_np(state_func(time)).squeeze()

            # Differenz (e = r - x)
            e01 = des_state - plant_state

            # Zustände des Reglers
            w = state[n:].T
            wd = controller_rhs(w, e01)

            # Ausgang des Reglers (Eingang der Strecke)
            v = orig_controller_output(w, e01)


            u = st.to_np(u_func(time)).squeeze() +   v

            args = np.concatenate([q, qd, u])
            qdd = qdd_fnc(*args.T)


            return np.concatenate([qd, qdd, wd])#*-


        def final_input_calculation(state, time):#*+
            """
            Funktion um nachträglich die wirksamen Stellgrößen zu berechnen
            (gleicher Code (Teilmenge) wie rhs, aber anderer Rückgabewert)
            """

            # Zustände der Strecke
            q = state[:n/2].T
            qd = state[n/2:n].T
            plant_state = np.concatenate([q, qd])

            # Soll-Zustand zur aktuellen Zeit:
            des_state = st.to_np(state_func(time)).squeeze()

            # Differenz (e = r - x)
            e01 = des_state - plant_state

            # Zustände des Reglers
            w = state[n:].T
            wd = controller_rhs(w, e01)

            # Ausgang des Reglers (Eingang der Strecke)
            v = orig_controller_output(w, e01)
            u = st.to_np(u_func(time)).squeeze() +   v
            return u

        # diese Funktion wird der rhs-Funktion als Attribut mitgegeben
        # => Fabrik-Funktion (create_simfunction) hat nur einen Rückgabewert:
        rhs.final_input_calculation = final_input_calculation
        return rhs #*-

    def apply_uncertainty(self, bound = 0.1, seed = 0):#*+-
        """
        Veränderung der Systemparameter
        (Abweichung zwischen Entwurfs- und Simulationsmodell)

        bound:    Schranke der relativen Abweichung
        seed:   Startwert für Zufallsgenerator
        """

        np.random.seed(seed) #*+
        Np = len(self.param_values)
        noise = np.random.rand(Np)*2-1 # zwischen -1 und 1
        rel_noise = 1+noise*bound # zwischen 1-bound und 1+bound

        keys, values = list(zip(*list(self.param_values.items())))
        new_values = np.array(values)*rel_noise
        self.param_values = dict(list(zip(keys, new_values))) #*-


# Regler-Matrix (Bk, Ak) laden und in Darstellung 1. Ordnung überführen

pfname = "data_feedback_matrix.pcl"
with open(pfname, "r") as pfile:
    pdict_cl = pickle.load(pfile)

full_fb_matrix  = pdict_cl['fb_matrix']
s = pdict_cl['s']

m = full_fb_matrix.shape[0]

# vorderen Teil abspalten
Bk = full_fb_matrix[:,:-m]
# hinteren Teil abspalten
Ak = full_fb_matrix[:,-m:]

p = Bk.shape[1]

state_feedback = sp.zeros(m, 2*p)
# vorderer Teil: Koordinaten -> Terme 0. Ordnung
state_feedback[:, :p] = Bk.subs(s,0)
# hinterer Teil: Geschwindigkeiten -> Terme 1. Ordnung
state_feedback[:, p:] = Bk.diff(s)

assert s not in state_feedback

# sympy Matrix -> Numpy array
Bk_state = st.to_np(state_feedback)#


# Systemgleichungen laden
pfname = "data_model_equations.pcl"
with open(pfname, "r") as pfile:
    pdict_eqn = pickle.load(pfile)
    print(pfname, "geladen")


tol = 1e-8 # Toleranz für den Integrator

t = sp.Symbol('t')

# Auswertezeit
# Simulationszeit

# Trajektoriendaten laden
traj_fname = "data_trajectories.pcl"
with open(traj_fname, "r") as pfile:
    pdict_ol = pickle.load(pfile)
    print(traj_fname, "geladen")

T_end = pdict_ol['T_end']
Tsim = T_end*1.8
tt = np.linspace(-1, Tsim, 1e3)

# symb. Trajektorienbeschreibung laden und in ausführbare Funktion umwandeln
u_traj = pdict_ol['u_traj']
u_func = st.expr_to_func(t, list(u_traj), eltw_vectorize=True)
state_traj = pdict_ol['state_traj']
state_func = st.expr_to_func(t, list(state_traj), eltw_vectorize=True)

# Instanz der Modelklasse erstellen: #*+
sim_mod = SimModel()

##-> Unbestimmtheiten berücksichtigen
#sim_mod.apply_uncertainty(bound = 0.05)

# rhs-Objekt auf Basis der Reglermatrizen und der veränderten Modell-Parameter
rhs = sim_mod.create_simfunction(Ak, Bk, s)

# Anfangswerte (laden und Anpassung an Zustandsdarstellung):
xa = list(pdict_ol['xa']) + [0,0,0] + [0]*sim_mod.number_of_controller_states
xa = st.to_np(xa).squeeze() # -> numpy array

##-> Anfangsfehler der Simulation vorgeben
#xa[0]+=.5

# Durchführung der eigentlichen Simulation
print("\n", "Simulation des geschlossenen Regelkreises", "\n")
res = odeint(rhs, xa, tt, rtol = tol, atol = tol)#*-

# Reglerzustände sind nicht relevant
r = res[:, :sim_mod.state_dim]
r2 = res[:, sim_mod.state_dim:]

x1, x2, x3, x4, x5, x6 = r.T

fname = "data_sim_results.txt"
np.savetxt(  fname, np.column_stack( (tt, r) )  )
print(fname, "geschrieben")
pl.grid(True)

savefig_flag = False

pl.plot(tt, x1, color = "0.0", label=r"$\varphi_0$") # schwarz
pl.plot(tt, x2, color = "0.2", label=r"$\varphi_1$") # dunkelgrau
pl.plot(tt, x3, color = "0.5", label=r"$\varphi_2$") # hellgrau

# Sollzustände
state_tt = np.array(state_func(tt)).T
X1, X2, X3, X4, X5, X6 = state_tt

pl.plot(tt, X1, ':', color = "0.0")
pl.plot(tt, X2, ':', color = "0.2")
pl.plot(tt, X3, ':', color = "0.5")
pl.legend(loc="best")
if savefig_flag:
    pl.savefig("simulation_x.pdf")


# Eingagssignale der Strecke:
u12 = np.zeros((len(tt), 2))
for i, (t_value, state_value) in enumerate(zip(tt, res)):
    u12[i,:] = rhs.final_input_calculation(state_value, t_value)

u1, u2 = u12.T
u1_des, u2_des = u_func(tt).T

pl.figure()
pl.plot(tt, u1_des, 'b:', label=r'$u_1^{\mathrm{soll}}$')
pl.plot(tt, u1, 'b-', label=r'$u_1$')
pl.plot(tt, u2_des, 'g:', label=r'$u_2^{\mathrm{soll}}$')
pl.plot(tt, u2, 'g-', label=r'$u_2$')
pl.title("$u_1$, $u_2$")
pl.legend(loc="best")
if savefig_flag:
    pl.savefig("simulation_u.pdf")

pl.show()


