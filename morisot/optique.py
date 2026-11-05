# -*- coding: utf-8 -*-
"""
Les briques communes : la pupille, la MFT (Matrix Fourier Transform), les masques échantillonnés, les
phases équivalentes à un masque décalé, le passe-bas du banc, les modes.

Conventions, valables dans tout Morisot :
    - une carte de phase est une OPD en nm, sur la grille de la pupille (N_PUP²) ;
    - x, y sont en unités de D, centrés entre les pixels ;
    - u, v sont en λ/D ; un mode à f c/p pose ses speckles à f λ/D.
"""
from functools import lru_cache #pour garder les calculs en memoire, lru pour last recently used
from math import factorial

import numpy as np
from scipy.ndimage import zoom

from . import config

#permet de figer le cache et de ne pas le modifier par accident
#source de bug lol
def _figer(tableau):
    """Rend un tableau en lecture seule : les caches ne doivent pas être modifiés."""
    tableau.flags.writeable = False
    return tableau

#on cree la pupille normalisé
def coordonnees(n=config.N_PUP):
    #retourne deux matrices de tailles 256x256 contenant les coordonne de chaque pixel en x et en y 
    # normalise!
    """x, y en unités de D, centrés entre les pixels : x va de −0,5 à 0,5."""
    centre = (n - 1) / 2.0
    y, x = (np.mgrid[:n, :n] - centre) / n
    return x, y


@lru_cache
def pupille(n=config.N_PUP):
    """Le disque de la pupille, n px de diamètre : le disque de commande du banc."""
    #on s'enrt selement pour simuler le SLM, chute raide, pour un bord smooth(pr zelda, coro, on utilise la fnc disque)
    x, y = coordonnees(n)
    return _figer((np.hypot(x, y) < 0.5).astype(float)) #on coverti en matrice de boolens 


@lru_cache

def zone(n=config.N_PUP):
    # on reproudit comment on mesure les rms sur le banc
#sur mithic on mesure l'opd sur un disque de diametre plus petit de 5 pixels pour eviter les effets de bords!!
#cf ce que Raphael m'avait dit et qu'il a ecrit ds son papier
    """Le disque où se lisent les rms : 5 px sous le bord, à l'échelle du banc."""
    x, y = coordonnees(n)
    return _figer(np.hypot(x, y) < 0.5 * config.RAYON_ZONE)


def disque(n, diametre_px, centre_x=0.0, centre_y=0.0, sur=8):
    """Un disque sur n×n px, chaque pixel pondéré par la part de sa surface couverte.

    Le centre est décalé de (centre_x, centre_y) px par rapport au centre de la
    grille. Le bord en niveaux de gris évite qu'un masque ne change par sauts
    d'un pixel.
    """
    milieu = (n - 1) / 2.0
    positions = (np.arange(n * sur) + 0.5) / sur - 0.5 - milieu
    yy, xx = np.meshgrid(positions - centre_y, positions - centre_x, indexing='ij')
    dedans = np.hypot(xx, yy) < diametre_px / 2.0
    return dedans.reshape(n, sur, n, sur).mean(axis=(1, 3))


def phase_rad(opd_nm):
    """L'OPD en nm convertie en phase, en radians."""
    return 2 * np.pi * opd_nm * 1e-9 / config.LAMBDA


# ----------------------------------------------------------------- la MFT

# porpagation entre plan de focal et pupille
#Pour le ZELDA et le coronographe on utilise la methode de babinet!!!!!!!! (cf coronographe.py, zelda.py)
#plus rapide et moins de cpu
def matrice_mft(n_pup, n_foyer, pas_ld):
    """La matrice A de la MFT, foyer centré sur l'axe, pas de `pas_ld` λ/D.

    E_foyer = A · E · Aᵀ / n_pup², la transformée de Fourier continue de la
    pupille avec dx = 1/n_pup ; à phase nulle, E_foyer(0) vaut l'aire de la
    pupille en unités de D².
    """
    x = (np.arange(n_pup) - (n_pup - 1) / 2.0) / n_pup # les coordonnes de la pupille normalisé
    u = (np.arange(n_foyer) - (n_foyer - 1) / 2.0) * pas_ld # les cooordonnes en plan focal
    #pas_ld permet de chosir l'echantillonage
    return np.exp(-2j * np.pi * np.outer(u, x))


def vers_foyer(champ, matrice):
    #champ := coresppond apupille ou le disque
    """La pupille vers le foyer."""
    n = champ.shape[-1] #on retrouve le nbr de pixel
    return matrice @ champ @ matrice.T / n ** 2


def vers_pupille(champ_foyer, matrice, pas_ld):
    """Le foyer vers la pupille : l'inverse continue, du = pas_ld."""
    return pas_ld ** 2 * (matrice.conj().T @ champ_foyer @ matrice.conj()) #en @D donc on fait ds les @directions


# ---------------------------------------------- les masques décalés
def tilt_nm(dx_ld, dy_ld, n=config.N_PUP):
    """L'OPD qui déplace la PSF de (dx, dy) λ/D."""
    #on ajt une carte de phase ur la pupille
    x, y = coordonnees(n)
    return config.LAMBDA * 1e9 * (dx_ld * x + dy_ld * y)


def defocus_nm(dz_mm, n=config.N_PUP):
    """L'OPD que voit un masque placé dz mm après le foyer : −dz/(8N²)·ρ².

    C'est la propagation de Fresnel du champ focal sur dz, écrite dans la
    pupille. À F/90, 1 mm donne 15,4 nm PV, soit 4,45 nm rms de Z4.
    """
    #on ajt une carte de phase ur la pupille
    x, y = coordonnees(n)
    w020_nm = dz_mm * 1e-3 / (8 * config.F_RATIO ** 2) * 1e9
    return -w020_nm * 4 * (x ** 2 + y ** 2)


@lru_cache
def phase_masque_nm(decalage, n=config.N_PUP):
    #permet de simuler l'effet sur la phase dans la pupille d'un decalage du masque
    """La phase à mettre devant le masque centré pour simuler le masque décalé.

    Un masque décalé de +s λ/D voit la PSF décalée de −s : un tilt de −s. Son
    déplacement en z est le défocus de `defocus_nm`. Derrière le masque, il faut
    appliquer l'opposé (le faisceau, lui, n'a pas bougé). Pour ZELDA cet opposé
    tombe dans le plan pupille de la caméra et ne change pas l'intensité ; pour
    le coronographe il tombe sur le Lyot et compte.
    """
    sx = decalage.dx_um / config.LAMBDA_F_UM
    sy = decalage.dy_um / config.LAMBDA_F_UM
    return _figer(tilt_nm(-sx, -sy, n) + defocus_nm(decalage.dz_mm, n))


# ----------------------------------------------------- les cartes
def depuis_banc(carte_banc):

    # carte banc: carte venant du banc en 418x418, que l'on met sur le SLM
    #convertut la carte du banc vers la grlle de Morisot
    """Une carte du banc (disque de commande de N_BANC px) sur la grille de Morisot.

    On garde le carré du disque de commande, centré, et on le rééchantillonne
    pour que le disque fasse exactement N_PUP px (cubique : la commande s'arrête
    à 75 c/p, bien sous les 128 c/p de la grille).
    """
    taille = carte_banc.shape[0]
    debut = int(round((taille - 1) / 2.0 - config.N_BANC / 2.0 + 0.5))
    carre = np.nan_to_num(np.asarray(carte_banc, float))[debut:debut + config.N_BANC,
                                                         debut:debut + config.N_BANC]
    sortie = zoom(carre, config.N_PUP / config.N_BANC, order=3, grid_mode=True, mode='grid-constant')
    return sortie[:config.N_PUP, :config.N_PUP] * pupille()


def noll_vers_nm(j):
    """L'indice de Noll j (1 = piston) en (n, m) : m > 0 pour cos(mθ), m < 0 pour sin."""
    n = int((-1 + np.sqrt(8 * (j - 1) + 1)) / 2)
    k = n % 2
    m = int((j - n * (n + 1) / 2 + k) / 2) * 2 - k
    if m != 0 and j % 2 == 1:
        m = -m
    return n, m


def zernike(j, n=config.N_PUP):
    """Le Zernike de Noll j sur la pupille, normalisé à 1 nm rms (Noll 1976), nul dehors.

    1 piston, 2 tip (x), 3 tilt (y), 4 défocus, 5-6 astigmatismes, 7-8 comas,
    9-10 trèfles, 11 sphérique.
    """
    ordre, m = noll_vers_nm(j)
    x, y = coordonnees(n)
    r, theta = 2 * np.hypot(x, y), np.arctan2(y, x)
    radial = sum((-1) ** k * factorial(ordre - k)
                 / (factorial(k) * factorial((ordre + abs(m)) // 2 - k) * factorial((ordre - abs(m)) // 2 - k))
                 * r ** (ordre - 2 * k) for k in range((ordre - abs(m)) // 2 + 1))
    if m == 0:
        carte = np.sqrt(ordre + 1) * radial
    elif m > 0:
        carte = np.sqrt(2 * (ordre + 1)) * radial * np.cos(m * theta)
    else:
        carte = np.sqrt(2 * (ordre + 1)) * radial * np.sin(-m * theta)
    return carte * pupille(n)


#fonction qui permet de trouver les coefs des modes choisis sur une carte
#on resoud par LMS
@lru_cache
def _projecteur(modes, n=config.N_PUP):
    #modes est une liste!!
    """La pseudo-inverse des Zernike `modes` (tuple d'indices de Noll) sur la pupille."""
    dedans = pupille(n) > 0
    base = np.stack([zernike(j, n)[dedans] for j in modes], 1)
    return _figer(np.linalg.pinv(base)), _figer(base)


def projeter(carte, modes, n=config.N_PUP):

    """Les coefficients (nm rms) des Zernike `modes` dans la carte, par moindres carrés."""
    projecteur, _ = _projecteur(tuple(modes), n)
    return projecteur @ carte[pupille(n) > 0]


def carte_zernike(coefficients, modes, n=config.N_PUP):
    """La carte Σ coefficients_k · Z_modes_k en nm, nulle hors de la pupille : l'inverse de `projeter`."""
    return sum(c * zernike(j, n) for c, j in zip(coefficients, modes))


def sans_modes(carte, modes, n=config.N_PUP):
    """La carte sans les Zernike `modes` (moindres carrés sur la pupille), nulle dehors."""
    projecteur, base = _projecteur(tuple(modes), n)
    dedans = pupille(n) > 0
    valeurs = carte[dedans]
    sortie = np.zeros_like(carte)
    sortie[dedans] = valeurs - base @ (projecteur @ valeurs)
    return sortie


def _projecteur_ptt(n=config.N_PUP):
    """La pseudo-inverse de piston, tip, tilt sur la pupille."""
    return _projecteur((1, 2, 3), n)


def sans_piston_tip_tilt(carte, n=config.N_PUP):
    """La carte sans piston ni tip/tilt (moindres carrés sur la pupille), nulle dehors."""
    projecteur, base = _projecteur_ptt(n)
    dedans = pupille(n) > 0
    valeurs = carte[dedans]
    sortie = np.zeros_like(carte)
    sortie[dedans] = valeurs - base @ (projecteur @ valeurs)
    return sortie


def rms(carte, n=config.N_PUP):
    """La rms du papier : piston et tip/tilt retirés, lue sur la zone."""
    return float(np.std(sans_piston_tip_tilt(carte, n)[zone(n)]))


def passe_bas(carte, coupure, n=config.N_PUP):
    """Le passe-bas de Hann du banc, coupure en c/p (MITHICstart, ncpa._filtre_passe_bas).

    Gain 0,5·(1 + cos(πf/fc)) jusqu'à fc, zéro au-delà. La carte est posée sur
    une grille deux fois plus grande pour que la FFT ne replie pas un bord de
    la pupille sur l'autre.
    """
    pup = pupille(n)
    grand = np.zeros((2 * n, 2 * n))
    grand[:n, :n] = carte * pup
    frequences = np.fft.fftfreq(2 * n) * n                 # c/p
    f = np.hypot(*np.meshgrid(frequences, frequences, indexing='ij'))
    fenetre = np.where(f <= coupure, 0.5 * (1 + np.cos(np.pi * f / coupure)), 0.0)
    return np.real(np.fft.ifft2(np.fft.fft2(grand) * fenetre))[:n, :n] * pup


def modes_fourier(f_min, f_max, n=config.N_PUP):
    """Les modes de Fourier aux fréquences entières f_min ≤ |f| ≤ f_max c/p.

    cos et sin(2π(fx·x + fy·y)) sur le demi-plan fx > 0 ou (fx = 0, fy > 0) :
    (fx, fy) et (−fx, −fy) donnent le même mode. Chacun vaut 1 nm rms sur la
    pupille, moyenne nulle. Rend (modes (n_modes, n, n), fréquences (n_modes, 2)).
    """
    x, y = coordonnees(n)
    pup = pupille(n)
    dedans = pup > 0
    borne = int(np.ceil(f_max))
    modes, frequences = [], []
    for fx in range(0, borne + 1):
        for fy in range(-borne, borne + 1):
            if fx == 0 and fy <= 0:
                continue
            if not f_min <= np.hypot(fx, fy) <= f_max:
                continue
            argument = 2 * np.pi * (fx * x + fy * y)
            for carte in (np.cos(argument), np.sin(argument)):
                carte = carte - carte[dedans].mean()
                modes.append(carte / carte[dedans].std() * pup)
                frequences.append((fx, fy))
    return np.array(modes), np.array(frequences, float)

#################### LE MIEUX EST QUE JE PRELEVE DIRECTEMENT LA DSP DU BANC EN FIN DE BOUCLE###########
def ecran(rms_nm, exposant=2.0, f_min=1.0, f_max=None, graine=0, n=config.N_PUP):
    """Un écran de phase aléatoire, DSP en f^−exposant entre f_min et f_max c/p.

    Tiré sur une grille deux fois plus grande (pas de périodicité visible),
    piston et tip/tilt retirés, mis à `rms_nm` sur la zone.
    """
    rng = np.random.default_rng(graine) #on choisit la methode d'aleatoire
    #avanatge, on peut reproudire. al'identique les nbrs aleatoires
    #suite de nmbr

    frequences = np.fft.fftfreq(2 * n) * n #frequences contenues 1D
    f = np.hypot(*np.meshgrid(frequences, frequences, indexing='ij')) #frequences contenues 2D

    f_max = n / 2.0 if f_max is None else f_max #frequnce maxinale par defaut, 128 cpp, c le plus approprié
    dans_bande = (f >= f_min) & (f <= f_max) #retourne un tableau de booleens
    #est ce que la frequence est comprise entre f_min et f_max?

    amplitude = np.where(dans_bande, np.where(f > 0, f, 1.0) ** (-exposant / 2.0), 0.0)

    #### DSP=amplitude**2
    bruit = rng.normal(size=f.shape) + 1j * rng.normal(size=f.shape)
    carte = np.real(np.fft.ifft2(bruit * amplitude))[:n, :n]
    carte = sans_piston_tip_tilt(carte, n)
    return carte * rms_nm / np.std(carte[zone(n)])
