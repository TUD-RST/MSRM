# coding: utf8

import sympy as sp
import numpy as np
import pickle

from control_aux import symb_tools as st

from IPython import embed as IPS

"""
 Skript zum Reglerentwurf für das Doppelpendel auf Rädern

"""

def calc_controller(Ap, Bp, desired_clcp, controller_variant = 1): #*+-
    """
    Vorgehen: Die Systemmatrix (Ap, -Bp) wird schrittweise mit Zusatzzeilen
    ergänzt, sodass die Linksteilerfreihheit erhalten bleibt.
    In der letzten Zeile wird ein polynomialer Ansatz für die Rückführung
    vorgegeben und die (parameterabhängige) Determinante bestimmt.
    Die Parameter werden dann aus dem Koeffizientenvergleich mit dem
    Wunsch-CLCP bestimmt.

    Der Ansatz wird derart vorgenommen, dass die resultirende Übertragungs-
    matrix
                Ap  -Bp
                Bk   Ak

    proper ist.
    """

    ApBp = Ap.row_join(-Bp)#*+
    p, m = Bp.shape

    # -> Entwurfsfreiheitsgrad

    if controller_variant == 1:

        # erste Ergänzungszeile (unten):
        # konkrete Wahl der Zahlen ist ein Entwurfsfreiheitsgrad
        Z3Z4_1 = sp.Matrix([0, 10, -10, 1, 0]).T

        # Symbole für Ansatz:
        N_symbols = 2*p + 1
        p_symbols = sp.symbols( 'p1:%i' % (N_symbols + 1) )

        P0 = sp.Matrix(p_symbols[:p]).T # für 0. Ordnung
        P1 = sp.Matrix(p_symbols[p:2*p]).T # für 1. Ordnung

        # zweite Zeile von Z3
        Z3_2 = (P0+P1*s)

        # zweite Zeile (nach hinten) ergänzen
        Z3Z4_2 = Z3_2.row_join(sp.Matrix([0, p_symbols[-1]+s]).T)
        hc_flag = True  #*-


    elif controller_variant == 2:

        Z3Z4_1 = sp.Matrix([-1000, 0, 0, -1000, -1000]).T
        #    Z3Z4_1 = sp.Matrix([10, 0, 0, 10, 10]).T


        # zweite Zeile direkt vorgeben:
        p1, p2, p3, p4, p5, p6, p7, p8, p9 = sp.symbols('p1:10')

        Z3Z4_2 = sp.Matrix([0, p1*s+p2, p3*s+p4, p5*s**2+p6*s+p7,
                                                 p8*s**2+p9*s+p7]).T

        p_symbols = list(st.atoms(Z3Z4_2, sp.Symbol))
        p_symbols.sort()
        p_symbols.remove(s)

        hc_flag = False

    elif controller_variant == 3:
        Z3Z4_1 = sp.Matrix([10, 0, 0, 10, 10]).T


        # zweite Zeile direkt vorgeben:
        p1, p2, p3, p4, p5, p6, p7, p8, p9 = sp.symbols('p1:10')

        Z3Z4_2 = sp.Matrix([0, p1+p2*s, p3+p4*s, p5*s**2+p6*s+p7,
                                                 p8*s**2+p9*s+p7]).T



        p_symbols = list(st.atoms(Z3Z4_2, sp.Symbol))
        p_symbols.sort()
        p_symbols.remove(s)

        hc_flag = False

    elif controller_variant == 4:
        # statischer Regler -> nicht proper
        Z3Z4_1 = sp.Matrix([0, -10, 10, 1, 0]).T


        # zweite Zeile direkt vorgeben:
        k1, k2, k3, k4, k5, k6, k7, k8, k9 = sp.symbols('k1:10')

        Z3Z4_2 = sp.Matrix([k1+k2*s, k3+k4*s, k5*s+k6, 0,1]).T



        p_symbols = list(st.atoms(Z3Z4_2, sp.Symbol))
        p_symbols.sort(key=str)
        p_symbols.remove(s)

        hc_flag = True

    else:
        raise ValueError("Unbekannte Regler-Variante")

    # Systemmatrix des geschlossenen Kreises (CLSM, Schritt 1): #*+
    CLSM1 = ApBp.col_join(Z3Z4_1)
    assert st.is_left_coprime(CLSM1)

    CLSM = CLSM1.col_join(Z3Z4_2)
    Z3Z4 = CLSM[p:, :] # = (Bk, Ak) (Zusatz-Zeilen)


    det = CLSM.berkowitz_det().expand()
    det = st.trunc_small_values(det)

    highest_coeff = st.poly_coeffs(det, s)[0]

    desired_clcp = sp.Poly(desired_clcp, s, domain = "EX")

    if hc_flag:
        # höchsten Koeffizienten angleichen (nicht immer notwendig)
        assert desired_clcp.is_monic
        desired_clcp *= highest_coeff

    # Determinante ist ein Polynom (s) mit den f-Param. in den Koeffizienten
    poly_det = sp.Poly(det, s, domain = "EX")
    deg = poly_det.degree()

    # Überprüfen, ob die Ordnung des Wunsch-CLCP mit der Ordnung
    # von CLSM.det() übereinstimmt
    assert desired_clcp.degree() == deg


    # Differenz soll identisch verschwinden
    # Koeff des Differenzpolynoms sollen all 0 sein
    diff_poly = st.trunc_small_values(poly_det - desired_clcp)


    # Koeff.Vergleich -> Gleichungssystem aufstellen
    # (durch den speziellen Ansatz linear in den Parametern)
    eqns = st.poly_coeffs(diff_poly, s)

    # Gleichungen (linke Seiten) nach 0 Auflösen
    sol = sp.solve(eqns, p_symbols)

    # sicherstellen, dass eine (eindeutige) Lösung gefunden wurde

    assert len(sol) == len(p_symbols)

    res = Z3Z4.subs(sol) # Endergebnis #*-

    # Ausgabe von BkAk:
    print("\n", "Reglermatrix BkAk:\n\n")
    sp.pprint(res)

    if 0:
        # optionale Überprüfung der Properness
        verify_properness(CLSM.subs(sol), deg)

    return res


def verify_properness(CLSM, deg):
        """
        CLSM: Systemmatrix des geschlossenen Kreises
        deg:  Grad der Determinante von CLSM
        -----

        Properness-Probe:
        Die Matrix des geschlossenen Kreises darf in keinem Eintrag einen
        höheren Grad im Zähler als im Nenner haben.

        Nenner ist überall das CLCP.
        Die Zähler ergeben sich aus CLSM.adj() * blockdiag(Ap, AK)
        """
        G2 = st.trunc_small_values(CLSM.adjugate().expand())

        # G3 = blockdiag(Ap, AK)
        # wird hier aus CLSM durch 0-setzen von Bp und Bk erzeugt
        G3 = CLSM*1 # Kopie
        G3[:3,3:] *=0
        G3[3:,:3] *=0

        G4 = G2*G3

        def deg_func(p):
            pp = sp.Poly(p, s, domain="EX")
            return pp.degree()

        G5 = st.trunc_small_values(G4.expand())

        max_num_degree = max(G5.applyfunc(deg_func))
        # Properness-Probe:
        if max_num_degree <= deg:

            print("Properness für alle Einträge erfüllt")
        else:
            print("Achtung: Properness nicht für alle Einträge erfüllt")


def roots_to_rpoly_expr(symb, *roots):
    """
    Hilfsfunktion.
    Wandelt die gegebene Liste von Nullstellen in einen reellen polynomialen
    Ausdruck um. Komplexe Nullstellen werden konjugiert komplex ergänzt.
    """
    assert len(roots) > 1
    all_roots = []
    for r in roots:
        all_roots.append(r)
        if not np.imag(r) == 0:
            all_roots.append(np.conjugate(r))


    assert len(all_roots) >= len(roots)
    res = 1
    for r in all_roots:
        res *= (symb - r)

    assert sp.I not in res.expand().atoms()

    return res

pfname = "data_model_equations.pcl"
with open(pfname, "r") as pfile:
    pdict = pickle.load(pfile)
    print(pfname, "geladen")

Ap = pdict['Ap']
Bp = pdict['Bp']
s = pdict['s']



clcp1a = (s+3)**7 #*+
clcp1b = roots_to_rpoly_expr(s, -1, -1.5+.5j, -2+1j, -2.5+2j)

clcp2a = roots_to_rpoly_expr(s, -.25, -2, -3, -3.1, -3.2, -10.3, -10.4, -10.5)
clcp2b = roots_to_rpoly_expr(s, -5.0, -2+1j, -2+1j, -2+1j, -10)
clcp2c = roots_to_rpoly_expr(s, -.25, -2, -20, -21, -22, -23, -24, -25) #[RL07]

clcp3a = roots_to_rpoly_expr(s, -.25, -2, -3, -3.1, -3.2, -10.3, -10.4, -10.5)
clcp3b = roots_to_rpoly_expr(s, -1.+1j, -2, -3+2j, -8+5j, -10)

clcp4a = (s+3)**6
clcp4b = (s+1)*(s+2)*(s+3)*(s+4)*(s+5)*(s+6) #*-

##-> Reglerstruktur und CLCP festlegen
BkAk = calc_controller(Ap, Bp, clcp4b, controller_variant = 4) #*+-


# Ergebnis in Datei schreiben

pdict = dict(fb_matrix = BkAk, s = s)
pfname = "data_feedback_matrix.pcl"
with open(pfname, "w") as pfile:
    pickle.dump(pdict, pfile)
    print(pfname, "geschrieben")

