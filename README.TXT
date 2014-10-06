Allgemeines
===========

Das Paket MSRM stellt Software-Routinen zur modellbasierten Steuerung und
Regelung eines mechanischen Systems zur Verfügung, am Beispiel eines von Natur
aus instabilen Fahrzeugs. Alle dafür notwendigen Schritte (Modellbildung,
Steuerungsentwurf (Planung einer Solltrajektorie), Reglerentwurf, Simulation
und Visualisierung) sind jeweils als einzelnes Python-Skript angelegt, die
unabhängig von einander ausgeführt und bearbeitet werden können.

Installationshinweise
=====================
Für die Ausführung der Skripte müssen folgende Sofware-Pakete installiert
sein:

python2.7.x (python3.x nicht getestet)
numpy
scipy
sympy
matplotlib

Auf Windows-Plattformen empfiehlt sich die Installation einer geeigneten python-
distribution, wie z.B.:
pythonxy, http://code.google.com/p/pythonxy/
winpython, http://code.google.com/p/winpython/

Schellstart
===========

Ausführung folgender Python-Skripte, unter Beachtung der Reihenfolge:

model.py
open_loop.py
closed_loop.py
simulation.py
visualization.py

Kontakt
=======
Fehler-Berichte, Verbesserungsvorschläge etc. bitte per Email an
Carsten.Knoll (ät) tu-dresden.de

