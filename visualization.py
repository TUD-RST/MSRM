# -*- coding: utf-8 -*-


"""
Skript zur schematischen Visualisierung der Bewegung des Doppelpendels auf
Rädern.
"""

import numpy as np
from numpy import sin, cos
import sympy as sp
import pickle
import time
import pylab as pl
from IPython import embed as IPS

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.path as mpath
Path = mpath.Path

from control_aux import symb_tools as st


# global colors:

dark = dict(facecolor="0.5", edgecolor ="0.0")
gray = dict(facecolor="0.8", edgecolor ="0.0")

def rotZ(x):
    """
    Rotationsmatrix in der Ebene
    """
    return np.array([cos(x), -sin(x), sin(x), cos(x)]).reshape(2,2)

def rect(x, y, wx, wy, phi = 0):
    pathdata = [
    (Path.MOVETO, (x, y)),
    (Path.LINETO, (x+wx, y)),
    (Path.LINETO, (x+wx, y+wy)),
    (Path.LINETO, (x, y+wy)),
    (Path.CLOSEPOLY, (x, y)),
    ]
    codes, verts = list(zip(*pathdata))

    verts = np.dot(rotZ(phi),np.array(verts).T).T

    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor='red', edgecolor='yellow', alpha=0.5)
    return patch

def roundrect(ax, x, y, z1, z2, w, h, phi = 0, **kwargs):
    k = 4.0/3
    pathdata = [
    (Path.MOVETO, (x, y)),
    (Path.CURVE4, (x, y-w/k)),
    (Path.CURVE4, (x+w, y-w/k)),
    (Path.CURVE4, (x+w, y)),
    (Path.LINETO, (x+w, y+h)),
    (Path.CURVE4, (x+w, y+h+w/k)),
    (Path.CURVE4, (x, y+h+w/k)),
    (Path.CURVE4, (x, y+h)),
#    (Path.LINETO, (x, y+wy)),
    (Path.CLOSEPOLY, (x, y)),
    ]
    codes, verts = list(zip(*pathdata))


    verts = np.array(verts).T
    verts[0, :]-=w/2.0 # correct x


    # Drehpunkt
    pivot = np.array([[x,y]]).T

    verts = np.dot(rotZ(phi), verts-pivot) + pivot
    verts = verts.T


    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, **kwargs)

    ax.add_patch(patch)

    # Befestigungspunkt
    r = w*0.25
    ax_point = np.dot(rotZ(phi), pivot.flatten() + np.r_[0,h] )
    #pl.plot(*zip(pivot, ax_point))
    patch1a = pl.Circle([z1,z2], r, **dark)
    ax.add_patch(patch1a)


def bell(ax, x, y, w1, w2, h, phi, **kwargs):
    """
    Zeichnet das glockenförmige obere Segment
    """
    k = 4.0/3
    k2 = 3.0
    pathdata = [
    (Path.MOVETO, (x, y)),
    (Path.CURVE4, (x, y-w1/k)),
    (Path.CURVE4, (x+w1, y-w1/k)),
    (Path.CURVE4, (x+w1, y)),
    # Ende des Kreises
    (Path.CURVE4, (x+w1, y+h/k2)),
    (Path.CURVE4, (x+w1+w2, y+h*(1-1/k2))),
    (Path.CURVE4, (x+w1+w2, y+h)),
    # Ende S-Kurve
    (Path.LINETO, (x-w2, y+h)),
    # Ende Waagerechte Linie
    (Path.CURVE4, (x-w2, y+h*(1-1/k2))),
    (Path.CURVE4, (x, y+h*1/k2)),
    (Path.CURVE4, (x, y)),
#    (Path.LINETO, (x, y+wy)),
    (Path.CLOSEPOLY, (x, y)),
    ]
    codes, verts = list(zip(*pathdata))

    verts = np.array(verts).T
    verts[0, :]-=w1/2.0 # correct x

    pivot = np.array([[x,y]]).T

    verts = np.dot(rotZ(phi), verts-pivot) + pivot
    verts = verts.T

    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, **kwargs)

    ax.add_patch(patch)





def bell2(ax, x, y, px, py, w1, w2, h, phi, **kwargs):

    x= px

    w1*=2

    r_ = np.r_
    c_ = np.c_
    d1 = 1
    d2 = 1
    d3 = .2
    d4 = .5
    d4L = .6

    d5 = .1
    d6 = .3
    d7 = .5+d4+d4L
    w2 = w1*1.1 # Breite oben
    p1 =r_[x, y]
    p2 =r_[x-d1, y-d2]

    p3_0 =r_[x+w1, y]
    dp3 =r_[d1, -d2]
    p3 = p3_0 + dp3
    p4 =r_[x+w1, y] # Ende ex-Kreis


    p5 = p3_0 - .1*dp3
    p6 =r_[x+w1, y+d4-d3]
    p7 =r_[x+w1, y+d4]

    p7b =r_[x+w1, y+d4+d4L]# Endpunkt Linie

    p8 =r_[x+w1, p7b[1]+d5] # Kontrollpunkt
    p9 =r_[x+w2, y+d7-d6]
    p10=r_[x+w2, y+d7] # obere rechte Ecke

    w3 = -(w2-w1)
    p11=r_[x+w3, y+d7] # obere linke Ecke

    p12=r_[x+w3, p9[1]] # Kontrollpunkt
    p13=r_[x, p8[1]] # Kontrollpunkt
    p14=r_[x, p7b[1]] # S-Kurve Endpunkt

    p14b=r_[x, p7[1]] # Linie-Endpunkt

    p15=r_[x, p6[1]] # senkrechter Konrollpunkt
    p16=r_[x, y] +.1*dp3*r_[1,-1] # schräger Konrollpunkt
    p17=r_[x, y] # schräger Konrollpunkt


    if 0:
        plot(p5, 'rx', ms = 5)
        plot(p6, 'rx', ms = 5)
        plot(p7, 'ro', ms = 5)
        plot(p7b, 'go', ms = 5)
        plot(p8, 'bo', ms = 5)
        plot(p9, 'mo', ms = 5)
        plot(p10, 'yo', ms = 5)

    pathdata = [
    (Path.MOVETO, p1),
    (Path.CURVE4, p2),
    (Path.CURVE4, p3),
    (Path.CURVE4, p4),

    (Path.CURVE4, p5),
    (Path.CURVE4, p6),
    (Path.CURVE4, p7),
    (Path.LINETO, p7b),

    (Path.CURVE4, p8),
    (Path.CURVE4, p9),
    (Path.CURVE4, p10),
    (Path.LINETO, p11),

    (Path.CURVE4, p12),
    (Path.CURVE4, p13),
    (Path.CURVE4, p14),

    (Path.LINETO, p14b),

    (Path.CURVE4, p15),
    (Path.CURVE4, p16),
    (Path.CURVE4, p17),

    (Path.CLOSEPOLY, (x, y))]


    codes, verts = list(zip(*pathdata))


    verts = np.array(verts).T
    verts[0, :]-=w1/2.0 # correct x



    pivot = np.array([[px,py]]).T


#    plot(pivot, 'ro', ms = 8)
#    plot([x,y], 'go', ms = 8)

    verts = np.dot(rotZ(phi), verts-pivot) + pivot
    verts = verts.T

    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, **dark)


    ax.add_patch(patch)

def wheel(ax, x, y, r, r0, phi, **kwargs):
    # von außen nach innen: a, b, c


    patch1a = pl.Circle([x,y], r, **dark)
    patch1b = pl.Circle([x,y], r*0.6, **gray)
    patch1c = pl.Circle([x,y], r*0.2, **dark)

    mp = np.dot(rotZ(phi), np.array([r-r0, 0])) + np.array([x,y])
    mp2 = np.dot(rotZ(phi), np.array([r, 0])) + np.array([x,y])



    patch2 = pl.Circle(mp, r0, **gray)

    ax.add_patch(patch1a)
    ax.add_patch(patch1b)
    ax.add_patch(patch1c)
    ax.add_patch(patch2)

    xx, yy = list(zip([x,y], mp2))
    # Linie statt Kreis.
    #pl.plot(xx, yy, 'k-')


def draw_unicycle(p0, p1, p2, colors):
    """
    :args:
        p0, p1, p2: Winkel
        colors:     Farb-Dictionary
    """

    dx = -.05*r
    dy =- .01*r
    r_rel = 1#.98

    rr = r

    mx0 = -p0*r
    my0 = r/2

    # Bezugspunkt des Lastbehälters ("Glocke")
    mx1, my1 = np.dot(rotZ(p1), np.array([0, l*1.2]))+np.array([mx0, my0])

    # Pivot-Punkt (Drehpunkt des Lastbehälters)
    mx1b, my1b = np.dot(rotZ(p1), np.array([0, l]))+np.array([mx0, my0])

    # Boden
    pl.plot([-10, 10], [-1,-1], 'k-')

    bell2(ax, mx1, my1, mx1b, my1b, b2, b3, l2, p2, **colors)
    roundrect(ax, mx0, my0, mx1b, my1b, b1, l, p1, **colors)
    wheel(ax, mx0, my0, rr, .05*r, p0,  **colors)

    # Zoom wiederherstellen
    pl.axis(my_axis)



def update_plot(axes):
    """
    Das Plot-Fenster mit neuen Daten ausstatten.
    Diese Funktion wird vom matplotlib-timer-Objekt aufgerufen.
    """
    axes.clear()

    i = C.i
    C.i += di  # globale Zählvariable erhöhen
    if C.i >= len(tt):
        time.sleep(2)
        C.i = 0

    t, p0, p1, p2 = phi_tt_sim[:, i]
    draw_unicycle(p0, p1, p2, colors1)

    # Ausgabe der aktuellen Zeit
    plt.text(0.06, 0.05, "t = %3.2fs" % t, transform = axes.transAxes)
    axes.figure.canvas.draw()

###########################################################################

colors1 = dict(facecolor="0.7", edgecolor="0.2", alpha=1)

# Simulationsergebnisse laden:
fname = "data_sim_results.txt"
phi_tt_sim = np.loadtxt(fname)[:, :4].T
tt = phi_tt_sim[0,:]

class Container:
    """
    Hilfslasse um global auf Variablen zugreifen zu können (als Attribute)
    """
    pass

C = Container()

C.i = 0 # globaler counter
di = 10 # Schritweite (Simulationsschritte pro frame)

# geometrische Parameter des Gefährts
r = 1
l = 2
b1 = 0.4
b2 = b1*2
b3 = b2/2
l2 = l*0.75


# Grafik-Fenster einrichten:

pl.rcParams['figure.subplot.bottom']=.1
pl.rcParams['figure.subplot.left']=.05
pl.rcParams['figure.subplot.top']=.98
pl.rcParams['figure.subplot.right']=.98

mm = 1./25.4 #mm to inch
scale = 3
fs = (85*mm*scale, 65*mm*scale)
fig = plt.figure(figsize = fs)
ax = fig.add_subplot(1, 1, 1)#
plt.axis('equal')
plt.axis([-3, 7, -1.6, 4.5])
my_axis = pl.axis() # zoom merken (wegen 'equal' nicht identisch mit Vorgaben)
ax.axis('off')


def live_animation():
    dt = .001  # in s
    interval=int(dt * 1000)  # interval in ms
    timer = fig.canvas.new_timer(interval=interval)

    timer.add_callback(update_plot, ax)
    timer.start()

    plt.show()

def gen_video_frames(t_, p0_, p1_, p2_, rate, single_frame=False):
    """
    Einzelne Bilder speichern.
    (Diese können dann zu einer Video-Sequenz zusammengefügt werden)
    """

    k =0
    dt = tt[1]-tt[0]

    # Index-Schritte pro Bild
    dk = int(1.0/rate * 1.0/dt)

    pic_nbr_offset = 10000 # für den Dateinamen
    pic_nbr = 1

    # ggf. pfad für Bilder anlegen:
    path = "video_trans4"
    import os
    if not os.path.exists(path):
        os.mkdir(path)

    while k < len(p0_):

        P0, P1, P2 =  list(zip(p0_, p1_, p2_))[k]
        t = t_[k]

        print(t)

        ax.clear()
        draw_unicycle(P0, P1, P2, colors1)
        plt.text(0.06, 0.05, "t = %3.2fs" % t, transform = ax.transAxes)

        pl.xticks([])
        pl.yticks([])


        pl.draw()

        if isinstance(single_frame, str):
            fname = single_frame
            pl.savefig(fname)
            print(fname, "gespeichert")
            break

        fname = "frame_%03d.jpg" % (pic_nbr + pic_nbr_offset)
        pl.savefig(os.path.join(path,fname))
        pic_nbr+=1

        k+=dk
        print("%i %%" % int(k *100/ len(tt)))


mod = 1
if mod == 1:
    live_animation()
elif mod == 2:
    # Einzelbilder Sequenz
    tt, phi0, phi1, phi2 = phi_tt_sim
    gen_video_frames(tt, phi0, phi1, phi2, 25)
else:
    # ein Einzelbild
    t0 = 3.3
    tt = phi_tt_sim[0,:]
    start_idx = np.where(tt>t0)[0][0]
    tt, phi0, phi1, phi2 = phi_tt_sim[:, start_idx:]
    gen_video_frames(tt, phi0, phi1, phi2, 25, single_frame="gefaehrt.pdf")







