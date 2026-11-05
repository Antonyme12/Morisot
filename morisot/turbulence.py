# -*- coding: utf-8 -*-
"""
La turbulence jouée par le SLM, et une pose longue sous turbulence.

Les écrans viennent de l'`Atmosphere` d'OOPAO (Héritier et al.,
github.com/cheritier/OOPAO), comme dans l'ancien jumeau : une couche sans
vent, de von Kármán (L0 = D), qui « bout » trame après trame,

    φ(t + dt) = ρ·φ(t) + √(1 − ρ²)·φ_neuf,    ρ = exp(−dt/τ),  dt = 1/f_slm,

le processus de `serie_turbulence` de MITHICstart. Le SLM du jumeau est parfait :
il joue l'écran tel quel, piston et tip/tilt retirés (du pointage, hors boucle
au banc). La turbulence s'ajoute à la voie commune : ZELDA et le coronographe la
voient tous les deux.

OOPAO s'importe ici seulement, depuis son clone (~/Documents/70_Logiciels/OOPAO,
installé en éditable) : le reste de Morisot ne dépend pas de lui.
"""
import contextlib
import io

import numpy as np

from . import config, controle, coronographe, optique

D_M = 8.0              # m : OOPAO veut des mètres ; seuls D/r0 et L0/D comptent
R0_DEPART = 0.15       # m à 500 nm : le r0 d'où part l'étalonnage, sans effet sur le résultat


def _silencieux():
    """Un contexte où ce qu'OOPAO imprime ne sort pas."""
    return contextlib.redirect_stdout(io.StringIO())


class Turbulence:
    """Des écrans de `rms_nm` (piston et tip/tilt retirés, sur la zone) qui se décorrèlent en `tau_s`.

    r0 est réglé une fois sur la moyenne de `n_echelle` écrans indépendants :
    pour une graine donnée l'écran vaut r0^(−5/6) fois un tirage fixe, d'où
    r0 = R0_DEPART × (rms_depart / rms_nm)^(6/5). Chaque écran fluctue ensuite
    autour de `rms_nm`.
    """

    def __init__(self, rms_nm, tau_s, f_slm=10.0, L0=D_M, n_echelle=10, graine_echelle=10):
        with _silencieux(), np.errstate(divide='ignore'):
            from OOPAO.Atmosphere import Atmosphere
            from OOPAO.Source import Source
            from OOPAO.Telescope import Telescope
            self.telescope = Telescope(resolution=config.N_PUP, diameter=D_M, samplingTime=1.0 / f_slm)
            etoile = Source('I', 0, display_properties=False)
            etoile * self.telescope
            # vent nul : OOPAO divise r0 par la vitesse moyenne pour son τ0, qu'on n'utilise pas
            self.atmosphere = Atmosphere(self.telescope, r0=R0_DEPART, L0=L0, windSpeed=[0.0],
                                         fractionalR0=[1.0], windDirection=[0.0], altitude=[0.0],
                                         t_boiling=tau_s)
            self.atmosphere.initializeAtmosphere(self.telescope)
            self.telescope + self.atmosphere
        rms_bruts = [optique.rms(self.nouvel_ecran(graine_echelle + k)) for k in range(n_echelle)]
        rms_depart = float(np.sqrt(np.mean(np.square(rms_bruts))))
        with _silencieux(), np.errstate(divide='ignore'):
            self.atmosphere.r0 = R0_DEPART * (rms_depart / rms_nm) ** (6.0 / 5.0)
        self.f_slm = f_slm
        self.tau_s = tau_s

    def changer_tau(self, tau_s):
        """Un autre temps de décorrélation, sans refaire l'atmosphère (lu à chaque `update`)."""
        self.atmosphere.layer_1.t_boiling = tau_s
        self.tau_s = tau_s

    def ecran(self):
        """L'écran en cours, en nm, sur la pupille, piston et tip/tilt retirés."""
        return optique.sans_piston_tip_tilt(np.array(self.telescope.OPD_no_pupil) * 1e9)

    def nouvel_ecran(self, graine):
        """Un écran neuf, tiré par cette graine ; le bouillonnement repart d'elle."""
        with _silencieux():
            self.atmosphere.generateNewPhaseScreen(seed=graine)
        return self.ecran()

    def suivant(self):
        """L'écran une trame du SLM plus tard."""
        with _silencieux():
            self.atmosphere.update()
        return self.ecran()


def pose_longue(banc, etat, turbulence, n_trames, zone_, graine=0, iefc=None, gain=0.5, transitoire_s=5.0):
    """Joue la turbulence trame par trame et intègre l'image coronographique.

    iefc : None pour laisser le dark hole figé ; sinon (matrice, modes, sondes,
    alpha) : l'iEFC tourne en continu, une image par trame. Il pose les sondes
    une à une (−s₁, +s₁, −s₂, +s₂, chacune sur la trame en cours, donc sur une
    turbulence qui a bougé), puis corrige ; la correction vaut à partir de la
    trame suivante. La matrice est celle mesurée sans turbulence.

    Le contraste de chaque trame est celui de l'image sans sonde : ce que le
    dark hole vaut à cet instant. La pose longue moyenne ces images après
    `transitoire_s`. Rend un dictionnaire : temps, contrastes instantanés,
    contraste en pose longue, image moyenne, rms de turbulence jouée.
    """
    commande = etat.commande
    if iefc is not None:
        matrice, modes, sondes, alpha = iefc
        inverse = controle.reconstructeur(matrice, alpha)
        poses = []
    premiere = int(round(transitoire_s * turbulence.f_slm))
    somme, n_images, contrastes, rms_turbulence = None, 0, [], []
    for trame in range(n_trames):
        ecran = turbulence.nouvel_ecran(graine) if trame == 0 else turbulence.suivant()
        rms_turbulence.append(optique.rms(ecran))
        phase = etat.statique + ecran - commande + etat.science
        image = coronographe.image(phase, banc)
        contrastes.append(coronographe.contraste(image, zone_))
        if trame >= premiere:
            somme = image.copy() if somme is None else somme + image
            n_images += 1
        if iefc is not None:
            numero_sonde, signe = divmod(len(poses), 2)
            sonde = sondes[numero_sonde]
            poses.append(coronographe.image(phase - sonde if signe == 0 else phase + sonde, banc)[zone_])
            if len(poses) == 2 * len(sondes):
                mesure = np.concatenate([poses[2 * p] - poses[2 * p + 1] for p in range(len(sondes))])
                commande = commande + gain * np.tensordot(-(inverse @ mesure), modes, axes=1)
                poses = []
    image_moyenne = somme / n_images
    return {'temps': np.arange(n_trames) / turbulence.f_slm,
            'contrastes': np.array(contrastes),
            'contraste_pose_longue': coronographe.contraste(image_moyenne, zone_),
            'image_moyenne': image_moyenne,
            'rms_turbulence': float(np.sqrt(np.mean(np.square(rms_turbulence))))}
