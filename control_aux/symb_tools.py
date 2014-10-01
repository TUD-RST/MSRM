# -*- coding: utf-8 -*-


"""

Dieses Modul beinhaltet eine Sammlung nützlicher Funktionen zur Vereinfachung
des symbolischen Rechnens basierend auf sympy.

Es ist eine Teilmenge der noch in Entwicklung befindlichen Toolbox

pycontroltools

siehe http://www.tu-dresden/rst/software
"""


import sympy as sp
import numpy as np

from collections import Counter


import random

import itertools as it


# convenience
np.set_printoptions(8, linewidth=300)

# avoid name clashes with sage
piece_wise = sp.functions.elementary.piecewise.Piecewise

t = sp.var('t')

zf = sp.numbers.Zero()

class equation(object):

    def __init__(self, lhs, rhs = 0):
        self.lhs_ = sp.sympify(lhs)
        self.rhs_ = sp.sympify(rhs)

    def lhs(self):
        return self.lhs_

    def rhs(self):
        return self.rhs_

    def __repr__(self):
        return "%s == %s" % (self.lhs_, self.rhs_)

    def subs(self, *args, **kwargs):
        lhs_  = self.lhs_.subs(*args, **kwargs)
        rhs_  = self.rhs_.subs(*args, **kwargs)
        return type(self)(lhs_, rhs_)


class Container(object):

    def __init__(self, **kwargs):
        assert len( set(dir(self)).intersection(kwargs.keys()) ) == 0
        self.__dict__.update(kwargs)


# Der Inhalt des quickfrac moduls
"""
quick access to the most useful functionality of the fractions module
"""

import fractions as fr

Fr = fr.Fraction

def fractionfromfloat(x_, maxden = 1000):
  """
  fraction from float
  args:
   x
   maxdenominator (default = 1000)
  """

  x = float(x_)
  assert x == x_ # fails intentionally for numpy.complex
  return Fr.from_float(x).limit_denominator(maxden)

def sp_fff(x, maxden):
    """ sympy_fraction from float"""
    return sp.Rational(fractionfromfloat(x, maxden))



def trans_poly(var, cn, left, right):
    """
    returns a polynomial y(t) that is cn times continous differentiable

    left and right are sequences of conditions for the boundaries

    left = (t1, y1,  *derivs) # derivs contains cn derivatives

    """
    assert len(left) == cn+2
    assert len(right) == cn+2

    t1, y1 = left[0:2]
    t2, y2 = right[0:2]

    assert t1 != t2

    for tmp in (y1, y2):
        assert not isinstance(tmp, (np.ndarray, np.matrix, sp.Symbol) )


    # store the derivs
    D1 = left[2:]
    D2 = right[2:]

    # preparations
    condNbr = 2 + 2*cn

    coeffs = map(lambda i: sp.Symbol('a%d' %i), range(condNbr))
    #poly =  (map(lambda i, a: a*var**i, range(condNbr), coeffs))
    #1/0
    poly =  sum(map(lambda i, a: a*var**i, range(condNbr), coeffs))

    Dpoly = map(lambda i: sp.diff(poly, var, i), range(1,cn+1))


    # create the conditions

    conds = []
    conds += [equation(poly.subs(var, t1) , y1)]
    conds += [equation(poly.subs(var, t2) , y2)]

    for i in range(cn):
        #

        conds += [equation(Dpoly[i].subs(var, t1) , D1[i])]
        conds += [equation(Dpoly[i].subs(var, t2) , D2[i])]


    sol = lin_solve_eqns(conds, coeffs)

    sol_poly = poly.subs(sol)

    return sol_poly


def symbs_to_func(expr, symbs, arg):
    """
    in expr replace x by x(arg)
    where x is any element of symbs
    """
    #TODO: assert all([isinstance(s, sp.Symbol) for s in symbs])
    funcs = [sp.Function(s.name)(arg) for s in symbs]

    return expr.subs(zip(symbs, funcs))


def jac(expr, *args):
    if not hasattr(expr, '__len__'):
        expr = [expr]
    return sp.Matrix(expr).jacobian(args)


def get_coeff_row(eq, vars):
    """
    takes one equation object and returns the corresponding row of
    the system matrix
    """
    if not isinstance(eq, equation):
        # assume its the lhs     and rhs = 0
        eq = equation(eq,0)

    if isinstance(vars, sp.Matrix):
        vars = list(vars)

    get_coeff = lambda var: sp.diff(eq.lhs(), var)
    coeffs =  map(get_coeff, vars)
    rest = eq.lhs() - sum([coeffs[i]*vars[i] for i in range( len(vars) )])
    coeff_row = map(get_coeff, vars) + [eq.rhs() - rest]
    return coeff_row

def lin_solve_eqns(eqns, vars):
    """
    takes a list of equation objects
    creates a system matrix of and calls sp.solve
    """
    n = len(eqns)

    vars = list(vars) # if its a vector
    m = len(vars)

    rows = [get_coeff_row(eq, vars) for eq in eqns]

    sysmatrix = sp.Matrix(rows)

    sol = sp.solve_linear_system(sysmatrix, *vars)

    return sol

# TODO: Doctest
def concat_cols(*args):
    """
    takes some col vectors and aggregetes them to a matrix
    """

    col_list = []

    for a in args:
        if a.shape[1] == 1:
            col_list.append( list(a) )
            continue
        for i in xrange(a.shape[1]):
            col_list.append( list(a[:,i]) )
    m = sp.Matrix(col_list).T

    return m

# other name:
col_stack = concat_cols


# TODO: Doctest
def concat_rows(*args):
    """
    takes some row (hyper-)vectors and aggregetes them to a matrix
    """

    row_list = []

    for a in args:
        if a.shape[0] == 1:
            row_list.append( list(a) )
            continue
        for i in xrange(a.shape[0]):
            row_list.append( list(a[i, :]) )
    m = sp.Matrix(row_list)

    return m

# other name:
row_stack = concat_rows

# geschrieben für Polynommatritzen
def col_minor(A, *cols, **kwargs):
    """
    returns the minor (determinant) of the columns in cols
    """
    n, m = A.shape

    method = kwargs.get('method', "berkowitz")

    assert m >= n
    assert len(cols) == n

    M = sp.zeros(n, n)
    for i, idx in enumerate(cols):
        M[:, i] = A[:, idx]

    return M.det(method = method).expand()


def general_minor(A, rows, cols, **kwargs):
    """
    selects some rows and some cols of A and returns the det of the resulting
    Matrix
    """

    method = kwargs.get('method', "berkowitz")

    Q = row_col_select(A, rows, cols)

    return Q.det(method = method).expand()


def all_k_minors(M, k, **kwargs):
    """
    returns all minors of order k of M

    Note that if k == M.shape[0]

    this computes all "column-minors"
    """
    m, n = M.shape

    assert k<= m
    assert k<= n

    row_idcs = list(it.combinations(range(m), k))
    col_idcs = list(it.combinations(range(n), k))

    rc_idx_tuples = list(it.product(row_idcs, col_idcs))

    method = kwargs.get('method', "berkowitz")

    res = []
    for rr, cc in rc_idx_tuples:
        res.append(general_minor(M, rr, cc, method = method))

    return res

def row_col_select(A, rows, cols):
    """
    selects some rows and some cols of A and returns the resulting Matrix
    """

    Q1 = sp.zeros(A.shape[0], len(cols))
    Q2 = sp.zeros(len(rows), len(cols))

    for i, c in enumerate(cols):
        Q1[:, i] = A[:, c]


    for i, r in enumerate(rows):
        Q2[i, :] = Q1[r, :]

    return Q2

def is_left_coprime(Ap, Bp=None, eps = 1e-10):
    """
    Test ob Ap,Bp Linksteilerfrei sind
    keine Parameter zulässig

    """

# folgendes könnte die Berechnung vereinfachen
#    r1, r2 = Ap.shape
#
#    assert r1 <= r2
#
#    minors = all_k_minors(Ap, r1)
#
#    minors = list(set(minors)) # make entries unique


    #print "Achtung, BUG: Wenn ein minor konstant (auch 0) ist kommt ein Fehler"
    r1, r2 = Ap.shape
    if Bp == None:
        # interpret the last columns of Ap as Bp
        Bp = Ap[:, r1:]
        Ap = Ap[:, :r1]
        r1, r2 = Ap.shape


    assert r1 == r2
    r = r1
    r1, m =  Bp.shape
    assert r1 == r
    assert m <= r

    M = (Ap*1).row_join(Bp)

    symbs = list(matrix_atoms(M, sp.Symbol))
    assert len(symbs) == 1
    symb = symbs[0]

    combinations = it.combinations(range(r+m), r)

    minors = [col_minor(M, *cols) for cols in combinations]

    nonzero_const_minors = [m for m in minors if (m !=0) and (symb not in m)]

    if len(nonzero_const_minors) > 0:
        return True

    #zero_minors = [m for m in minors if m == 0]

    # polymionors (rows belong together)
    all_roots = [roots(m) for m in minors if symb in m]

    # obviously all minors where zeros
    if len(all_roots) == 0:
        return False

    # in general the arrays in that list have differnt lenght
    # -> append a suitable number of roots at inf

    max_len = max([len(a) for a in all_roots])
    root_list = [np.r_[a, [np.inf]*(max_len-len(a))] for a in all_roots]

    all_roots = np.array(root_list)

    # now testing if some finite root is common to all minors
    for i in range(all_roots.shape[0]):
        test_roots = all_roots[i, :]

        other_roots = np.delete(all_roots, i, axis = 0)

        for tr in test_roots:
            if tr == np.inf:
                continue

            min_dist = np.min(np.abs(other_roots-tr), axis = 1)
            if np.all(min_dist < eps):
                # the test root was found in all other minors

                print "critical root:", tr
                return False

    return True


def get_expr_var(expr, var = None):
    """
    auxillary function
    if var == None returns the unique symbol which is contained in expr:
    if no symbol is found, returns None
    """
    expr = sp.sympify(expr)
    if not var == None:
        assert isinstance(var, sp.Symbol)
        return var
    else: # var == None
        symbs = list(expr.atoms(sp.Symbol))
        if len(symbs) == 0:
            return None
        elif len(symbs) == 1:
            return symbs[0]
        else:
            errmsg = "%s contains more than one variable: %s " % (expr, symbs)
            raise ValueError, errmsg


def poly_degree(expr, var=None):
    """
    returns degree of monovariable polynomial
    """
    var = get_expr_var(expr, var)
    if var == None:
        return sp.sympify(0)

    P = sp.Poly(expr, var, domain = "EX")
    return P.degree()


def poly_coeffs(expr, var=None):
    """
    returns all (monovariate)-poly-coeffs (including 0s) as a list
    first element is highest coeff.
    """
    var = get_expr_var(expr, var)
    if var == None:
        return [expr]

    P = sp.Poly(expr, var, domain="EX")

    pdict = P.as_dict()

    d = P.degree()

    return [pdict.get((i,), 0) for i in reversed(xrange(d+1))]


def coeffs(expr, var = None):
    # TODO: besser über as_dict
    # TODO: überflüssig wegen poly_coeffs?
    """if var == None, assumes that there is only one variable in expr"""
    expr = sp.sympify(expr)
    if var == None:
        vars = filter(lambda a:a.is_Symbol, list(expr.atoms()))
        if len(vars) == 0:
            return [expr] # its a constant
        assert len(vars) == 1
        var=vars[0]
        dom = 'RR'
    else:
        dom = 'EX'
    return sp.Poly(expr, var, domain =dom).all_coeffs()

arr_float = np.frompyfunc(np.float, 1,1)


def to_np(arr, dtype=np.float):
    """ converts a sympy matrix in a nice numpy array
    """
    if isinstance(arr, sp.Matrix):
        symbs = list(matrix_atoms(arr, sp.Symbol))
        assert len(symbs) == 0, "no symbols allowed"

    # because np.int can not understand sp.Integer
    # we temporarily convert to float
    # TODO: make this work with complex numbers..
    arr1 = arr_float( np.array(arr) )
    return np.array( arr1, dtype )

def roots(expr):
    import scipy as sc
    return sc.roots(coeffs(expr))

def real_roots(expr):
    import scipy as sc
    r = sc.roots(coeffs(expr))
    return np.real( r[np.imag(r)==0] )

def zeros_to_coeffs(*z_list, **kwargs):
    """
    calculates the coeffs corresponding to a poly with provided zeros
    """

    s = sp.Symbol("s")
    p = sp.Mul(*[s-s0 for s0 in z_list])

    real_coeffs = kwargs.get("real_coeffs", True)
    c = np.array(coeffs(p, s), dtype=np.float)

    if real_coeffs:
        c = np.real(c)
    return c


def rev_tuple(tup):
    return [(t[1], t[0]) for t in tup]


def matrix_atoms(M, *args, **kwargs):
    sets = [m.atoms(*args, **kwargs) for m in list(M)]
    S = set().union(*sets)

    return S


def atoms(expr, *args, **kwargs):
    if isinstance(expr, (sp.Matrix, list)):
        return matrix_atoms(expr, *args, **kwargs)
    else:
        return expr.atoms(*args, **kwargs)


def trunc_small_values(expr, lim = 1e-10, n=1):
    expr = ensure_mutable( sp.sympify(expr) )

    a_list = list(atoms(expr, sp.Number))
    subs = []
    for a in a_list:
        if sp.Abs(a) < lim:
            subs.append((sp.Abs(a), 0))
            # substituting Abs(a) circumvents Problems in terms like sin(...)
            if a < 0:
                subs.append((a, 0))

    res = expr.subs(subs)

    if n <= 1:
        return res
    else:
        return trunc_small_values(res, lim, n-1)



def clean_numbers(expr, eps=1e-10):
    """
    trys to clean all numbers from numeric noise
    """

    if isinstance(expr, (list, tuple)):
        return [clean_numbers(elt, eps) for elt in expr]

    expr = trunc_small_values(expr)

    maxden = int(1/eps)
    floats = list(atoms(expr, sp.Float))
    rats = []
    dummy_symbs = []
    symb_gen = sp.numbered_symbols('cde', cls = sp.Dummy)
    for f in floats:
        rat = sp_fff(f, maxden)
        rats.append(rat)
        dummy_symbs.append(symb_gen.next())

    res1 = expr.subs(zip(floats, dummy_symbs))
    res2 = res1.subs(zip(dummy_symbs, rats))

    return res2


def zip0(xx, arg = 0):
    """ useful shortcut for substituting equilibrium points
    example: zip0([x1,x2,x3]) -> [(x1, 0), (x2, 0), (x3,0)]"""

    return zip(xx, [arg]*len(xx))


def aux_make_tup_if_necc(arg):
    """
    checks whether arg is iterable.
    if not return (arg,)
    """
    if not hasattr(arg, '__len__'):
        return (arg,)

    return arg


def expr_to_func(args, expr, modules = 'numpy', **kwargs):
    """
    wrapper for sympy.lambdify to handle constant expressions
    (shall return a numpyfied function as well)

    this function bypasses the following problem:

    f1 = sp.lambdify(t, 5*t, modules = "numpy")
    f2 = sp.lambdify(t, 0*t, modules = "numpy")

    f1(np.arange(5)).shape # -> array
    f2(np.arange(5)).shape # -> int


    Some special kwargs:
    np_wrapper == True:
        the return-value of the resulting function is passed through
        to_np(..) before returning

    """

    # TODO: sympy-Matrizen mit Stückweise definierten Polynomen
    # numpy fähig (d.h. vektoriell) auswerten


    # TODO: Unittest


    # TODO: only relevant if numpy is in modules

    expr = sp.sympify(expr)
    expr = ensure_mutable(expr)
    expr_tup = aux_make_tup_if_necc(expr)
    arg_tup = aux_make_tup_if_necc(args)

    new_expr = []
    arg_set = set(arg_tup)
    for e in expr_tup:
        assert isinstance(e, sp.Expr)
        # args (Symbols) which are not in that expression
        diff_set = arg_set.difference(e.atoms(sp.Symbol))

        # add and subtract the respective argument such that it occurs
        # without changing the result
        for d in diff_set:
            assert isinstance(d, sp.Symbol)
            e = sp.Add(e, d, -d, evaluate = False)

        new_expr.append(e)

    if not hasattr(expr, '__len__'):
        assert len(new_expr) == 1
        new_expr = new_expr[0]


    # extract kwargs specific for lambdify
    printer = kwargs.get('printer', None)
    use_imps = kwargs.get('use_imps', True)
    func = sp.lambdify(args, new_expr, modules, printer, use_imps)




    if kwargs.get('np_vectorize', False):
        func1 = np.vectorize(func)
    else:
        func1 = func

    if kwargs.get('special_vectorize', False):
        def func2(*allargs):
            return to_np(func(*allargs))

        f = np.float
        func3 = np.vectorize(func2, otypes = [f,f,f, f,f,f])
        return func3

    if kwargs.get('eltw_vectorize', False):
        # elementwise vectorization to handle piecewise expressions
        assert len(new_expr) >=1
        funcs = []
        for e in new_expr:
            func_i = sp.lambdify(args, e, modules, printer, use_imps)
            func_iv = np.vectorize(func_i)
            funcs.append(func_iv)

        def func2(*allargs):
            # each result should be a 1d- array
            results = [to_np(f(*allargs)) for f in funcs]

            # transpose, such that the input axis (e.g. time) is the first one
            return to_np(results).T.squeeze()

        return func2

    if kwargs.get('np_wrapper', False):
        def func2(*allargs):
            return to_np(func1(*allargs))
    elif kwargs.get('list_wrapper', False):
        def func2(*allargs):
            return list(func1(*allargs))
    else:
        func2 = func1
    return func2


def ensure_mutable(arg):
    """
    ensures that we handle a mutable matrix (iff arg is a matrix)
    """
    # TODO: e.g. sp.sympify converts a MutableMatrix to ImmutableMatrix
    # maybe this changes in future sympy releases
    # which might make this function obsolete (?)
    if isinstance(arg, sp.matrices.MatrixBase):
        return as_mutable_matrix(arg)
    else:
        return arg




def as_mutable_matrix(matrix):
    """
    sympy sometimes converts matrices to immutable objects
    this can be reverted by a call to    .as_mutable()
    this function provides access to that call as a function
    (just for cleaner syntax)
    """
    return matrix.as_mutable()

def is_col_reduced(A, symb, return_internals = False):
    """
    tests whether polynomial Matrix A is column-reduced

    optionally returns internal variables:
        the list of col-wise max degrees
        the matrix with the col.-wise-highest coeffs (Gamma)

    Note: concept of column-reduced matrix is important e.g. for
    solving a Polynomial System w.r.t. highest order "derivative"

    Note: every matrix can be made col-reduced by unimodular transformation
    """
    Gamma = as_mutable_matrix(A*0)
    n, m = A.shape

    assert n == m

    A = trunc_small_values(A)

    # degrees:
    A_deg = to_np(matrix_degrees(A, symb), dtype = np.float)
    max_degrees = list(A_deg.max(axis=0)) # columnwise maximum

    # TODO: unit-Test
    # handle zero columns:
    infty = float(sp.oo)
    max_degrees = [int(md) if not md == -infty else md for md in max_degrees]

    # maximum coeffs:
    for j in range(m):
        deg = max_degrees[j]
        for i in range(n):
            Gamma[i,j] = get_order_coeff_from_expr(A[i,j], symb, deg)

    result = Gamma.rank() == m
    if return_internals:
        # some functions might need this information
        internals = Container(Gamma = Gamma, max_degrees = max_degrees)
        return result, internals
    else:
        return result

def is_row_reduced(A, symb, *args, **kwargs):
    """
    transposed Version of is_col_reduced(...)
    """
    res = is_col_reduced(A.T, symb, *args, **kwargs)
    if isinstance(res, tuple):
        C = res[0]
        C.Gamma = C.Gamma.T
    return res


def get_col_reduced_right(A, symb, T = None, return_internals = False):
    """
    Takes a polynomial matrix A(s) and returns a unimod Transformation T(s)
    such that   A(s)*T(s) (i.e. right multiplication) is col_reduced.

    Approach is taken from appendix of the PHD-Thesis of S. O. Lindert (2009)

    :args:
        A:  Matrix
        s:  symbol
        T:  unimod-Matrix from preceeding steps

    -> recursive approach

    :returns:
        Ar: reduced Matrix
        T:  unimodular transformation Matrix


    This code is based on appendix A1 of '2009: Lindert, Sven-Olaf,
    Dissertation, TU Dresden'
    """

    n, m = A.shape
    assert n == m

    if T == None:
        T = sp.eye(n)
    else:
        assert T.shape == (n, m)
        d = T.berkowitz_det().expand()
        assert d != 0 and not symb in d


    A_work = trunc_small_values(sp.expand(A*T))


    cr_flag, C = is_col_reduced(A_work, symb, return_internals = True)

    # C.Gamma is the matrix with col-wise highest coeff
    if cr_flag:
        # this is the only exit point
        res = A_work.expand(), T
        if return_internals:
            res += (C,)
        return res
    else:
        pass
        # C.Gamma is nonregular

    g = C.Gamma.nullspace()[0]
    non_zero_cols_IDX = to_np(g).flatten() != 0
    # get the max_degrees wrt. to each non-zero component of g
    non_zero_cols_degrees = to_np(C.max_degrees)[non_zero_cols_IDX]

    N = max(non_zero_cols_degrees)
    # construct the diagonal matrix
    diag_list = []
    for i in range(m):
        cd = col_degree(A_work[:, i],symb)
        diag_list.append( symb**int(N-cd) )

    # gamma_p:
    gp = sp.diag(*diag_list)*g


    T1 = unimod_completion(gp, symb)

    TT = trunc_small_values( sp.expand(T*T1) )

    # recall this method with new T

    return get_col_reduced_right(A, symb, TT, return_internals)


def get_order_coeff_from_expr(expr, symb, order):
    """
    example:
        3*s**2 -4*s + 5, s, 3 -> 0
        3*s**2 -4*s + 5, s, 2 -> 3
        3*s**2 -4*s + 5, s, 1 -> -4
        3*s**2 -4*s + 5, s, 9 -> 0
    """
    p = sp.Poly(expr, symb, domain = "EX")
    default = 0
    return p.as_dict().get( (order,), default )


def element_deg_factory(symb):
    """
    returns a function for getting the polynomial degree of an expr. w.r.t.
    a certain symbol
    """
    def element_deg(expr):
        return poly_degree(expr, symb)

    return element_deg


def matrix_degrees(A, symb):

    element_deg = element_deg_factory(symb)

    return A.applyfunc(element_deg)


def col_degree(col, symb):
    return max(matrix_degrees(col, symb))


def unimod_completion(col, symb):
    """
    takes a column and completes it such that the result is unimodular
    """

    # there must at least one nonzero constant in col:

    n, m = col.shape
    assert m == 1
    element_deg = element_deg_factory(symb)

    idx = None
    for i, c in enumerate(list(col)):
        if c != 0 and element_deg(c) == 0:

        # we want the index of the first non-zero const. of col
            idx = i
            break

    assert not idx == None, "there should have been a nonzero const."


    T = sp.eye(n)

    T[:, idx] = col

    return T


def do_laplace_deriv(laplace_expr, s, t):
    """
    Example:
    laplace_expr = s*(t**3+7*t**2-2*t+4)
    returns: 3*t**2  +14*t - 2
    """

    if isinstance(laplace_expr, sp.Matrix):
        return laplace_expr.applyfunc(lambda x: do_laplace_deriv(x, s,t))

    exp = laplace_expr.expand()

    #assert isinstance(exp, sp.Add)

    P = sp.Poly(exp, s, domain = "EX")
    items = P.as_dict().items()

    res = 0
    for key, coeff in items:
        exponent = key[0] # exponent wrt s

        res += coeff.diff(t, exponent)

    return res
