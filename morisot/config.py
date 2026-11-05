# -*- coding: utf-8 -*-
"""
Les constantes de MITHiC parfait, et les décalages de ses masques.

Tout nombre qui vient du banc est ici, avec sa source. Les autres modules
n'en ont aucun en dur.
"""
from dataclasses import dataclass #creeer des classes pour stocker des valeurs plus simplement que ds un codde normal

from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]     # la racine du dépôt, au-dessus de morisot/
DONNEES = RACINE / 'donnees'                     # la commande du banc, seule donnée importée
RESULTATS = RACINE / 'resultats'                 # les figures (plot=True), hors du dépôt : voir .gitignore

# --- la source et le foyer des masques --------------------------------------
LAMBDA = 670.7e-9                        # m, la diode du banc
F_RATIO = 90                             # le foyer de ZELDA et du coronographe (MITHIC.ini ; Pourcelot 2021)
# a reverifier idealement
#taille physique d'un lambda/D
LAMBDA_F_UM = LAMBDA * F_RATIO * 1e6     # 60,36 µm : un λ/D dans le plan des masques

# --- ZELDA from MITHIC.ini de pyzelda ------------------------------------------
ZELDA_DIAMETRE_M = 64e-6                 # 1,06 λ/D
ZELDA_PROFONDEUR_M = 0.350e-6            # silice fondue

# --- le coronographe selon Pourcelot et al --------------
FPM_DIAM_LD = 5.38                       # λ/D, masque focal opaque
LYOT_RATIO = 0.77                        # diamètre du Lyot rapporté à la pupille

# --- les grilles --------------------------------------------------------------
N_PUP = 256                              # px sur le diamètre de la pupille : Nyquist à 128 c/p
N_BANC = 418                             # px, le disque de commande du SLM (commande_banc.npy : 418 px)
RAYON_ZONE = (N_BANC / 2 - 5) / (N_BANC / 2)   # les rms se lisent 5 px sous le bord, comme au banc
Q = 4                                    # px par λ/D dans l'image science
DEMI_CHAMP_LD = 12                       # λ/D, la demi-largeur de l'image science
N_FOYER = 2 * int(round(DEMI_CHAMP_LD * Q)) + 1    #taille de k'image science
PX_MASQUE = 128                          # px sur le diamètre de chaque masque focal
#ce qui donne un pas d'echantillonage pour le ZELDA de 1.06/128=0.008 lambda/D par pixel


# --- la boucle NCPA du banc , valeur par defaut seulement ----
#     journal du 04/09 : « 15 it @ 75 c/p, gain 1.0 », depuis un SLM à zéro
NCPA_N_ITER = 15
NCPA_GAIN = 1.0
NCPA_PASSE_BAS = 75                      # c/p

# --- le dark hole, valer par defaut uniquemnt -------------------------------------------------------------
R_IN, R_OUT = 3.0, 8.0                   # λ/D, le demi-anneau
MARGE_MODES = 1.5                        # c/p de part et d'autre du demi-anneau pour les modes
GAIN_EFC = 0.8

#le decalage du zelda et du coro
@dataclass(frozen=True)
class Decalage:
    """Le déplacement d'un masque focal par rapport au foyer nominal.

    dx_um, dy_um : dans le plan du masque, en µm (un λ/D vaut 60,36 µm).
    dz_mm : le long de l'axe, en mm, positif quand le masque s'éloigne de la pupille.
    """
    dx_um: float = 0.0
    dy_um: float = 0.0
    dz_mm: float = 0.0

#le decalage du lyot
@dataclass(frozen=True)
class DecalageLyot:
    """Le déplacement du Lyot, en fraction du diamètre de la pupille.

    Pas de dz : la propagation sur dz demande le diamètre du faisceau au Lyot,
    qu'aucun document du banc ne donne.
    """
    dx: float = 0.0
    dy: float = 0.0


@dataclass(frozen=True)
class Banc:
    """Où sont les trois masques. Banc() : tous à leur place."""
    zelda: Decalage = Decalage()
    fpm: Decalage = Decalage()
    lyot: DecalageLyot = DecalageLyot()
