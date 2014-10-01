# coding: utf8
from __future__ import absolute_import

import sympy as sp
from sympy import pi, Matrix
import pylab as pl
import numpy as np
import pickle

from control_aux import symb_tools as st
from IPython import embed as IPS


u"""
Skript zum Steuerungsentwurf für das Doppelpendel auf Rädern

"""


# Systemgleichungen laden:
pfname = "data_model_equations.pcl"
with open(pfname, "r") as pfile:
    pdict = pickle.load(pfile)
    print pfname, "geladen"



# Systemgleichungen und Symbole in den globalen Namensraum einfügen
globals().update(pdict)

F = Matrix(sp.symbols('F0:3'))


# Keys haben Datentyp str
numparams = dict(m0 = 1, m1 = 1, m2 = 1, r = 1, g = 10, l1 = 2, l2 = 0.3,
                 s1 = 1, s2 = 1, J0 = 0.1, J1 = 0.1, J2 = 0.1,
                 delta0 = .0, delta1 = .1, delta2 = .1)

# keys: Symbole
params_symb_keys = dict([(par, numparams.get(par.name)) for par in pdict['params'] ])

# Immutable -> Mutable
M0 = Matrix(pdict['M0'])
K0 = Matrix(pdict['K0'])
Bp = Matrix(pdict['Bp'])
D0 = Matrix(pdict['D0'])

# Laplace-Variable
s = sp.Symbol('s')

Ap = (M0*s**2 + D0*s+K0).subs(numparams)

ABp = Ap.row_join(Bp)

# Linksteilerfreihheit überprüfen (Spaltennummerierung beginnt bei 0) #*+
S1 = set(st.roots(st.col_minor(ABp, 0,1,2))) # Nullstellen des OLCP
S2 = set(st.roots(st.col_minor(ABp, 2,3,4)))
assert len(S1.intersection(S2)) == 0 #*-
# Schnittmenge leer -> Minoren haben keine gemeinsame Nullstelle
# -> Linksteilerfrei


##-> Festlegung der Basisgrößen: #*+
# Ergänzung der System-Matrix, sodass diese quadratisch und unimodular wird

k1, k2 = sp.symbols('k1, k2')

# Betrachtung von drei Varianten:
# Ansatz1: xi1 := k1*phi0 + k2*phi1,    xi2 := phi2
# Ansatz2: xi1 := k1*phi0 + k2*phi2,    xi2 := phi1
# Ansatz3: xi1 := -phi0,                xi2 := k1*phi1 + k2*phi2

# Randbedingungen am Anfang
xa = Matrix( [0, 0, 0])
# RB Ende:
xb = Matrix( [-4, 0, 0])

variant = 1
if variant == 1:
    Z = Matrix([[k1, k2, 0, 0, 0], [0, 0, 1, 0 ,0]])
    # res = {k1: 0.0333333333333333, k2: 0.0474178403755869}
    T_end = 4.25
elif variant == 2:
    Z = Matrix([[k1, 0, k2, 0, 0], [0, 1, 0, 0 ,0]])
    # res = {k1: -0.100000000000000, k2: -0.0577464788732394}
    T_end = 9.5
elif variant == 3:
    Z = Matrix([[-1, 0, 0, 0, 0], [0, k1, k2, 0 ,0]])
    # res = {k1: -0.100000000000000, k2: -0.0577464788732394}
    T_end = 17
    xb = Matrix( [2, 0, 0])
else:
    raise ValueError, "Unerwartete Variante"

M = st.row_stack(ABp, Z) # Hyper-Zeilen zusammenfügen #*-
M = st.clean_numbers(M).as_mutable()

# Forderung: k1, k2 so wählen, dass det == 1
# Liste aller Koeff. des Polynoms M.det()-1
det = st.trunc_small_values(M.berkowitz_det().expand())
eqns = sp.Poly(det-1, s, domain = 'EX').as_dict().values()
eqns = st.clean_numbers(eqns)
# alle müssen identisch 0 werden
res = sp.solve(eqns, [k1,k2])

M = M.subs(res)

assert M.det().expand() == 1  # Konsistenzprüfung auf Unimodularität

U_R = M.adjugate()  # Hier inv == adjugate (weil det == 1) #*+
U_12R = U_R[:3, 3:]
U_22R = U_R[3:, 3:] #*-



##-> Wunschtrajektorien im Zeitbereich festlegen #*+

Z1 = M[-2:, :3]

# Definition der Basisgrößen Xi aus Systemgrößen X:
# Xi := Z1 * X

xi_a = Z1 * xa
xi_b = Z1 * xb

# Übergangspolynome (Trajektorien der Basissignale (Zeitbereich))
t = sp.Symbol('t')
xi_polys = []
##-> Glattheitsanforderung (>=3) ist ein Entwurfsfreiheitsgrad
cn = 3 # Glattheitsforderung (legt Anzahl der Randbed. fest)

for i in range(2):

    # Randbedingungen:
    left = (0,xi_a[i,0]) + (0,)*cn
    right = (T_end,xi_b[i,0]) + (0,)*cn

    poly = st.trans_poly(t, cn, left, right) # Polynome bestimmen
    print "xi_{0}(t) = ".format(i), poly.evalf()

    # Stückweise definierte Funktion für konstante Teile am Anfang und Ende:
    pw = sp.Piecewise((left[1], t<left[0]), (poly, t<T_end), (right[1], True))
    xi_polys.append(pw)

# Liste in Matrix umwandeln
xi_traj = sp.Matrix(xi_polys) #*-

xi_traj = st.trunc_small_values(xi_traj) # numerisches Rauschen beseitigen



# Trajektorien der Winkel (Systemgrößen): #*+
# ("Gemischte Darstellung": Laplace-Bereich und Zeitbereich)
PHI = U_12R*xi_traj

# Laplace-Variable als Ableitungsoperator anwenden:
phi_traj = st.do_laplace_deriv(PHI, s, t)

#  Ausdrücke in ausführbare Funktion umwandeln
xi_func = st.expr_to_func(t, list(xi_traj), eltw_vectorize=True)
phi_func = st.expr_to_func(t, list(phi_traj), eltw_vectorize=True)# *-

tt = np.linspace(-1, T_end*1.5, 1e3)

phi_tt = phi_func(tt).T
xi1, xi2 = xi_func(tt).T

# Schalter für das Speichern von Grafiken
savefig_flag = False

# Darstellung der Verläufe der Basisignale
pl.plot(tt, xi1, 'k-', label =  r"$\xi_1(t)$")
pl.plot(tt, xi2, 'k--', label =  r"$\xi_2(t)$")
pl.legend(loc="best")
pl.title(ur'Verläufe von $\xi_1, \xi_2$')
if savefig_flag:
    pl.savefig("xi_t.pdf")


phi0, phi1, phi2 = phi_tt # Komponenten extrahieren

pl.figure()
pl.plot(tt, phi0, 'k-', label =  r"$\varphi_0(t)$")
pl.plot(tt, phi1, 'k--', label = r"$\varphi_1(t)$")
pl.plot(tt, phi2, 'k:', label = r"$\varphi_2(t)$")
pl.xlabel("t")
pl.title(ur'Verläufe von $\varphi_0, \varphi_1, \varphi_2$')
pl.legend(loc="best")

if savefig_flag:
    pl.savefig("phi_t.pdf")


phi0 = np.array(phi0)
phi1 = np.array(phi1)
phi2 = np.array(phi2)


# Stellgrößen berechnen

U = U_22R*xi_traj
u_traj = st.do_laplace_deriv(U, s, t)
u_func = st.expr_to_func(t, list(u_traj), eltw_vectorize=True)
u_tt = u_func(tt).T
u1, u2 = u_tt

pl.figure()
pl.plot(tt, u1, 'k-', label=  "$u_1(t)$")
pl.plot(tt, u2, 'k--', label= "$u_2(t)$")
pl.xlabel("t")
pl.legend(loc="best")
pl.title(ur'Verläufe von $u_1, u_2$')
if savefig_flag:
    pl.savefig("u_t.pdf")


# Solltrajektorien für Zustände und Stellgrößen als Ausdrücke in t speichern

phi_d_traj = phi_traj.diff(t)
state_traj = phi_traj.col_join(phi_d_traj)

pdict = dict(state_traj = state_traj, u_traj = u_traj, xi_traj = xi_traj,
             T_end = T_end, xa = xa, xb = xb, param_values = params_symb_keys)

#traj_fname = "data_trajectories%i.pcl" %variant
traj_fname = "data_trajectories.pcl"
with open(traj_fname, "w") as pfile:
    pickle.dump(pdict, pfile)
    print traj_fname, "geschrieben"

if 1:
    pl.show()

