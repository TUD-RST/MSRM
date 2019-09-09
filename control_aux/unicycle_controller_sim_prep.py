# coding: utf8
#from __future__ import absolute_import

import sympy as sp
from sympy import cos, sin, pi, Matrix
import numpy as np
from scipy.integrate import odeint
import sys


from . import symb_tools as st
#from ipHelp import IPS, ST, ip_syshook, dirsearch, sys, TracerFactory
import pylab as pl

import pickle
import os


# TODO: -> aux-Paket?


"""
Modul um den Regler, der als Ak(s)*V(s)+Bk(s)*E(s) = 0 gegeben ist
in Zustandsdarstellung zu überführen.

Vorgehen:
    * es müssen neue Zustandsvariablen eingeführt werden.
    * grad( Ak.det() ) legt ihre Anzahl fest
    * Die entsprechenden Gleichungen ergeben sich aus dem Auflösen
    obiger Gleichung nach der höchsten Ableitung von v, d.h. nach
    der spaltenweise-höchsten auftretenden s-Potenz.
    * Wenn Ak, nicht spaltenreduziert ist, kann nach den höchsten Abl.
    nicht aufgelöst werden.

    => Koordinatentransformation:
    (Ak * T) * (T.inv * V) =: Ak_ * W
    Ak_  ist spaltenreduziert -> Auflösung möglich.
    Nach der Berechnung von w(t) (durch Integration)
    kann v(t) aus V = T * W berechnet werden.
"""


def generate_state_matrices(Ak, Bk, symb):
    """
    Erstellt aus dem polynomial gegebenen lin. dyn. System
    Ak(s)*V(s) + Bk(s)*E(s)
    ein lineares dynamisches System in Zustandsdarstellung

    w_dot = A*w+ B*e
    y = C*w + D*e

    und eine Matrix T_ mit der die Ursprünglichen Größen aus den
    neuen Systemgrößen (Zuständen und Eingängen) berechnet werden können.

    v = T_ * (w.T, e.T).T
    """

    res = st.get_col_reduced_right(Ak, symb, return_internals = True)
    Ak_, T, C = res

    ww = sp.Matrix( sp.symbols("w1:%i" %(Ak_.shape[1]+1)) )
    ee = sp.Matrix( sp.symbols("e1:%i" %(Bk.shape[1]+1)) )


    s = symb
    s_power_list = [s**d for d in C.max_degrees]

    Gamma_s = C.Gamma * sp.diag(*s_power_list)
    # Matrix ohne die höchsten s-Potenzen
    Lambda = Ak_ - Gamma_s

    state_chains = []
    state = []
    for k, d in enumerate(C.max_degrees):
        chain = []
        for i in range(d):
            w = sp.Symbol( 'w%i_%i' %(k+1, i) )
            chain.append(w)
            state.append(w)
        state_chains.append(chain)


    rhs = -C.Gamma.inv()* ( Lambda*ww + Bk*ee )


    # Ersetzen der Terme s**3*w1 durch w1_3  etc.
    subslist = []
    for k, chain in enumerate(state_chains):
        for i, chain_w in enumerate(chain):
            original_w = ww[k]
            subslist.append( (s**(i)*original_w, chain_w) )

    # Ersetzen der Terme s*e2 durch e1_2
    # (Trennung von abgeleiteten und nicht abgeleiteten Größen)

    assert max(list(st.matrix_degrees(Bk, symb))) == 1

    ee0 = list( sp.symbols("e0_1:%i" %(Bk.shape[1]+1)) )
    ee1 = list( sp.symbols("e1_1:%i" %(Bk.shape[1]+1)) )


    for e, e0, e1 in zip(ee, ee0, ee1):
        subslist.append((e, e0))
        subslist.append((s*e, e1))

    # Liste Umkehren, damit Ersetzung vom höchsten zum niedrigsten durchläuft
    subslist.reverse()

    rhs2 = sp.expand(rhs).subs(subslist)

    # Ausgangsgleichung
    z_lhs_list = [] # "left hand side"
    z_eq_list = []
    y_lhs_list = []
    y_eq_list = []

    for k, chain in enumerate(state_chains):
        y_lhs_list.append(ww[k])
        if len(chain) == 0:
            # für diese Komponente wird nicht integriert
            y_eq_list.append(rhs2[k])
        else:
            # Ausgangs-Ende der Kette ist teil von y
            y_eq_list.append(chain[0])

            # definitorische Gleichungen
            z_eq_list.extend(chain[:-1])
            # Ableitung des letzten Zustandes aus rhs
            z_eq_list.append(rhs2[k])

            for i in range(1, len(chain)+1):
                z_lhs_list.append(s**i*ww[k])

    z_vector = sp.Matrix(z_eq_list)
    y_vector = sp.Matrix(y_eq_list)

    A = z_vector.jacobian(state)
    B = z_vector.jacobian(ee0+ee1)
    C = y_vector.jacobian(state)
    D = y_vector.jacobian(ee0+ee1)


    # jetzt Transformation auf originale Koordinaten:
    # V = T*W

    T_subslist = list(zip(z_lhs_list, z_eq_list))
    T_subslist.reverse()
    T_subslist.extend( list(zip(y_lhs_list, y_eq_list)) )

    # TODO: kann man zeigen dass T*W sich immer aus dem Zustand und den
    # Eingängen bestimmen lässt?

    v_eq_list = sp.expand(T * ww).subs(T_subslist)
    v_vector = sp.Matrix(v_eq_list)


    # in v_vector dürfen nur noch Zustände und Eingänge vorkommen
    assert st.atoms(v_vector, sp.Symbol).issubset(state + ee0 + ee1)



    T_state_input = v_vector.jacobian(state + ee0 + ee1)
    assert len( st.atoms(T_state_input, sp.Symbol) ) == 0

    # TODO: Eigentlich braucht man den Ausgang y nicht.


    return A, B, C, D, T_state_input



def poly_matr_to_state_funcs(Ak, Bk, symb):
    """
    Diese Funktion erstellt für das durch Ak, Bk gegebene Polynomiale System

    Ak(s)*V + Bk(s)*E = 0

    zwei Funktionen:

    a) eine, welche die rechte Seite der Zustandsdarstellung bildet:
    w_dot = rhs(w, e)

    b) eine, welche die ursprüngliche Ausgangsgröße aus dem Zustand
    und dem Eingang bestimmt.

    Für den Fall dass Ak nicht von s abhängt, kann v direkt bestimmt werden.
    Dann hat w die Dimension 0.
    """

    if symb not in st.atoms(Ak):
        # Regler hat keinen dynamischen Anteil

        assert not Ak.det() == 0
        Ak_inv = Ak.inv()

        def state_rhs(*args):
            return []

        state_rhs.state_dim = 0

        B1 = st.to_np(-Ak_inv * Bk.subs(symb, 0))
        B2 = st.to_np(-Ak_inv * Bk.diff(symb))
        assert symb not in B2

        m, p = Bk.shape

        def orig_output(w, e01):
            v = np.dot(B1, e01[:p]) + np.dot(B2, e01[p:])
            return v

        return state_rhs, orig_output



    state_matrices = generate_state_matrices(Ak, Bk, symb)
    A, B, C, D, T_state_input = [st.to_np(M) for M in state_matrices]

    n1, n2 = A.shape

    assert n1 == n2

    T1 = T_state_input[:, :n1]
    T2 = T_state_input[:, n1:]

    def state_rhs(w, e01):
        w_dot = np.dot(A, w) + np.dot(B, e01)
        return w_dot

    def orig_output(w, e01):
        v = np.dot(T1, w) + np.dot(T2, e01)
        return v

    # Diese beiden Funktionen werden jetzt zurückgegeben:

    state_rhs.state_dim = n2

    return state_rhs, orig_output



if __name__ == "__main__":

    # Regler-Matrix (Bk, Ak) laden und an Zustandsraumdarstellung anpassen

    #pfname = "uc_feedback_matrix_static.pcl"
    pfname = "uc_feedback_matrix.pcl"
    with open(pfname, "r") as pfile:
        pdict = pickle.load(pfile)

    full_fb_matrix  = pdict['fb_matrix']
    s = pdict['s']

    m = full_fb_matrix.shape[0]

    # vorderen Teil abspalten
    Bk = full_fb_matrix[:,:-m]
    # hinteren Teil abspalten
    Ak = full_fb_matrix[:,-m:]

    Ak = sp.Matrix([[10, 10], [s**2+ 3*s + 4, 2*s**2+s+4]])



    f1, f2 = poly_matr_to_state_funcs(Ak, Bk, s)

