# -*- coding: utf-8 -*-

import sympy as sp
import pickle
from sympy import Matrix

from IPython import embed as IPS
from control_aux.model_tools import generate_model, Rz
from control_aux.symb_tools import symbs_to_func, zip0

"""
Skript zur Herleitung der Modellgleichungen für das einachsige Doppelpendel
auf Rädern
"""
# Variable für Zeit
t = sp.Symbol('t')

# Parameter-Symbole
params = sp.symbols('l1, l2, s1, s2, delta0, delta1, delta2,'
                    'J0, J1, J2, m0, m1, m2, r, g')
l1, l2, s1, s2, delta0, delta1, delta2, J0, J1, J2, m0, m1, m2, r, g = params

# Modellparameter-Werte:
numparams = dict(m0=1, m1=1, m2=1, r=1, g=10, l1=2, l2=0.3,
                 s1=1, s2=1, J0=0.1, J1=0.1, J2=0.1,
                 delta0=.0, delta1=.1, delta2=.1)

# Symbole für Winkel (absolut, d.h. auf Umgebung bezogen)
q = Matrix(sp.symbols("q0:3"))
q = symbs_to_func(q, q, t)

q0, q1, q2 = q

# Winkelgeschwindigkeit
qd = q.diff(t)
q0d, q1d, q2d = qd

# Hilfsrößen für die Geometrie-Beschreibung:

# Einheitsvektoren
ex = Matrix([1, 0])
ey = Matrix([0, 1])

M0 = Matrix([-r * q0, r])  # Mittelpunkt des Rades

S1 = M0 + Rz(q1) * ey * s1
S2 = M0 + Rz(q1) * ey * l1 + Rz(q2) * ey * s2

M0d = M0.diff(t)
S1d = S1.diff(t)
S2d = S2.diff(t)

# kinetische Energie
T_rot = ( J0 * q0d ** 2 + J1 * q1d ** 2 + J2 * q2d ** 2 ) / 2
T_trans = ( m0 * M0d.T * M0d + m1 * S1d.T * S1d + m2 * S2d.T * S2d ) / 2

T = T_rot + T_trans[0]

# potentielle Energie
V = m1 * g * S1[1] + m2 * g * S2[1]

# verallgemeinerte Kräfte sollen in den Gelenken (d.h. relativ) wirken
# das System ist aber in absoluten Koordinaten modelliert,
# d.h. der Algorithmus in generate_model interpretiert die Kraft-Symbole
# ebenfalls als absolut -> Umrechnung notwendig

# verallgemeinerte Absolut-Kräfte
# (gegenüber der Umgebung, also bezogen auf absolute Koord.)
F = Matrix(sp.symbols('Fp0:3'))

# (verallgemeinerte) Gelenkg-Kräfte (bezogen auf relative Koordinaten)
F_rel = Matrix(sp.symbols('F0:3'))

# Permutations-Matrix (invertiert)
P = Matrix([1, 0, 0, 1, 1, 0, 1, 1, 1]).reshape(3, 3).inv()
F_expr = P.T * F_rel

# Datenstruktur mit Modellgleichungen erzeugen
sys_model = generate_model(T, V, q, F)  # *+-

# Absolut-("Pseudo")-Kräfte durch Gelenkkräfte ersetzen
sys_model.substitute_ext_forces(F, F_expr, F_rel)

# Dissipationskräfte einführen:
Delta = delta0, delta1, delta2
qd_relative = [sys_model.qds[0], sys_model.qds[1] - sys_model.qds[0],
               sys_model.qds[2] - sys_model.qds[1]]
diss_subslist = \
    [(F_rel[i], F_rel[i] - Delta[i] * qd_relative[i]) for i in range(3)]
sys_model.eq_list = sys_model.eq_list.subs(diss_subslist)

# Faktor -1 weil Kraftrichtung entgegen Koordinatenrichtung eingeführt
sys_model.eq_list = \
    sys_model.eq_list.subs([(F_rel[i], -F_rel[i]) for i in range(3)])

# Lagrange-Gleichungen sind aufgestellt.
# -> Modell umformen (nach Winkelbeschleunigungen auflösen, etc)

print("lineares Modell bestimmen")
qs = sys_model.qs
qds = sys_model.qds

# Ausgabe der Bewegungsgleichungen
qdds = sys_model.qdds

eqns = sys_model.eq_list
eqns_a = eqns.subs(g, 0).subs(zip0(Delta) + zip0(F_rel))
eqns_b = eqns - eqns_a
C = eqns_a.subs(zip0(qdds))

# Massematrix bestimmen
M = (eqns_a - C).jacobian(qdds)
M.simplify()
C.simplify()

# Gleichgewichtslage festlegen: (Winkel und Geschwindigkeiten = 0) #*+
x0 = zip0(qs) + zip0(qds)

sys_model.M0 = M.subs(x0)
sys_rest = sys_model.eq_list.subs(zip0(sys_model.qdds))
sys_model.K0 = sys_rest.jacobian(qs).subs(x0)


# Matrix der Dissipationsterme:
sys_model.D0 = sys_model.eq_list.jacobian(sys_model.qds).subs(x0)

# Eingangsmatrix (hier noch allgemein)
sys_model.B0 = sys_model.eq_list.jacobian(F_rel)

sys_model.params = params

# lineares Polynom-Matrix-Modell:
s = sp.Symbol('s')
sys_model.s = s
sys_model.Ap =\
    (sys_model.M0 * s ** 2 + sys_model.D0 * s + sys_model.K0).subs(numparams)

# Unteraktuiertes System F0 = 0 -> nur die letzten 2 Spalten von B nutzen
sys_model.Bp = sys_model.B0[:, 1:]
#sp.pprint((sys_model.Ap.row_join(sys_model.Bp))) #*-


sys_model.numparams = numparams


# Vorbereiten zum serialisieren
pdict = {}
for k in ['qs', 'qds', 'qdds', 'M0', 'K0', 'D0', 'B0', 'params',
          'extforce_list', 'Ap', 'Bp', 's', 'numparams']:
    pdict[k] = getattr(sys_model, k)

if 1:
    # zeitaufwendiger Block ("if 0:" beschleunigt debugging)
    print("nichtlineares Modell aufbereiten (zeitaufwendig)")

    M = sp.trigsimp(M.expand())
    d = M.det()
    d = sp.trigsimp(d, method="fu")

    print("adjungierte Massenmatrix berechnen und vereinfachen")
    adj = M.adjugate()

    adj = sp.trigsimp(adj)

    Minv = adj / d
    tmp_rhs = sp.trigsimp(sys_model.eq_list.subs(zip0(sys_model.qdds)))

    sys_model.rhs = -Minv * sys_model.eq_list.subs(zip0(sys_model.qdds))
    pdict['rhs'] = sys_model.rhs

# nichtlineares und lineares Modell in eine Datei schreiben
pfname = "data_model_equations.pcl"
with open(pfname, "wb") as pfile:
    pickle.dump(pdict, pfile)
    print(pfname, "geschrieben")



