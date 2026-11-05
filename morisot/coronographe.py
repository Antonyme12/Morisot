# -*- coding: utf-8 -*-
"""
Le coronographe de Lyot du banc : masque focal opaque de 5,38 λ/D, Lyot de 0,77 D.

    E_lyot = E − TF⁻¹[M·TF(E)]       (Babinet : on retire ce que le masque arrête)
    E_image = TF[L · E_lyot]

Le terme retiré ne vit que dans le masque, d'où une petite grille focale très
fine pour lui. Masque focal décalé : la phase de `optique.phase_masque_nm`
devant, son opposée derrière, qui tombe sur le Lyot.

Contraste : |E|² rapporté au pic de la PSF sans masque focal, Lyot nominal en
place, sans aberration. C'est une constante : elle ne dépend pas de la phase.
"""
from functools import lru_cache

import numpy as np

from . import config, optique


@lru_cache
def _matrices(n=config.N_PUP):
    """Les MFT vers le masque focal (fine) et vers l'image science (Q px/λD)."""
    pas_masque = config.FPM_DIAM_LD / config.PX_MASQUE
    n_bloc = config.PX_MASQUE + 2
    vers_masque = optique._figer(optique.matrice_mft(n, n_bloc, pas_masque))
    masque = optique._figer(optique.disque(n_bloc, config.PX_MASQUE))
    vers_image = optique._figer(optique.matrice_mft(n, config.N_FOYER, 1.0 / config.Q))
    return vers_masque, masque, pas_masque, vers_image


@lru_cache
def lyot(decalage=config.DecalageLyot(), n=config.N_PUP):
    """Le diaphragme de Lyot, éventuellement décalé, sur la grille de la pupille."""
    return optique._figer(optique.disque(n, config.LYOT_RATIO * n, decalage.dx * n, decalage.dy * n))


@lru_cache
def pic(n=config.N_PUP):
    """Le pic de la PSF sans masque focal, Lyot nominal, sans aberration."""
    *_, vers_image = _matrices(n)
    champ = optique.vers_foyer(optique.pupille(n) * lyot(n=n), vers_image)
    return float(np.max(np.abs(champ) ** 2))


def propager(champ_pupille, banc=config.Banc(), n=config.N_PUP):
    """Le champ dans l'image science, en racine de contraste : linéaire en champ_pupille."""
    vers_masque, masque, pas_masque, vers_image = _matrices(n)
    devant = np.exp(1j * optique.phase_rad(optique.phase_masque_nm(banc.fpm, n)))
    champ = champ_pupille * devant
    champ_lyot = champ - optique.vers_pupille(masque * optique.vers_foyer(champ, vers_masque),
                                              vers_masque, pas_masque)
    champ_lyot = champ_lyot * devant.conj() * lyot(banc.lyot, n)
    return optique.vers_foyer(champ_lyot, vers_image) / np.sqrt(pic(n))


def champ(phase_nm, banc=config.Banc(), n=config.N_PUP):
    """Le champ coronographique (N_FOYER², complexe) pour une phase en nm."""
    return propager(optique.pupille(n) * np.exp(1j * optique.phase_rad(phase_nm)), banc, n)


def _log10(image_contraste, plancher=1e-12):
    """log10 de l'image, les pixels sous `plancher` ramenés au plancher (pas de −inf)."""
    return np.log10(np.clip(image_contraste, plancher, None))


def image(phase_nm, banc=config.Banc(), n=config.N_PUP, logscale=False):
    """L'image coronographique, en contraste ; logscale=True la rend en log10."""
    sortie = np.abs(champ(phase_nm, banc, n)) ** 2
    return _log10(sortie) if logscale else sortie


def psf(phase_nm, banc=config.Banc(), n=config.N_PUP, logscale=False):
    """L'image sans masque focal, Lyot en place, en contraste (pic 1 sans aberration) ;
    logscale=True la rend en log10."""
    *_, vers_image = _matrices(n)
    champ_lyot = optique.pupille(n) * np.exp(1j * optique.phase_rad(phase_nm)) * lyot(banc.lyot, n)
    sortie = np.abs(optique.vers_foyer(champ_lyot, vers_image)) ** 2 / pic(n)
    return _log10(sortie) if logscale else sortie


def coordonnees_foyer():
    """u, v en λ/D sur l'image science, (0, 0) sur le pixel central."""
    milieu = (config.N_FOYER - 1) / 2.0
    v, u = (np.mgrid[:config.N_FOYER, :config.N_FOYER] - milieu) / config.Q
    return u, v


def zone(r_in=config.R_IN, r_out=config.R_OUT, cote=+1):
    """Le demi-anneau r_in ≤ r < r_out λ/D, côté u > 0 (cote=+1), u < 0 (−1), ou entier (0).

    Un correcteur de phase seul crée des speckles conjugués : creuser en (u, v)
    rallume (−u, −v). D'où un demi-anneau, sauf aberrations d'amplitude nulles.
    """
    u, v = coordonnees_foyer()
    r = np.hypot(u, v)
    dans_anneau = (r >= r_in) & (r < r_out)
    if cote == 0:
        return dans_anneau
    return dans_anneau & ((u > 0) if cote > 0 else (u < 0))


def contraste(image_contraste, zone_):
    """Le contraste moyen sur la zone."""
    return float(np.mean(image_contraste[zone_]))


def contraste_radial(image_contraste, r_in=0.0, r_out=config.DEMI_CHAMP_LD, pas=0.25, cote=0):
    """Le profil de contraste : le contraste moyen d'anneaux de `pas` λ/D, de r_in à r_out.

    cote comme `zone` (0 : anneaux complets). Rend (rayons, contrastes), deux tableaux
    (n_anneaux,) : le milieu de chaque anneau en λ/D, et son contraste moyen (nan si vide).
    Sous 1/Q = 0,25 λ/D, les anneaux n'ont presque plus de pixels.
    """
    bords = np.arange(r_in, r_out, pas)
    contrastes = []
    for r in bords:
        anneau = zone(r, r + pas, cote)
        contrastes.append(contraste(image_contraste, anneau) if anneau.any() else np.nan)
    return bords + pas / 2, np.array(contrastes)


def gain(image_reference, image_contraste, zone_):
    """Le gain en contraste sur la zone : contraste de la référence / contraste de l'image.

    Plus grand que 1 : l'image est plus sombre que la référence. Par exemple la PSF
    (référence) contre l'image coronographique, ou l'image avant contre après l'EFC.
    """
    return contraste(image_reference, zone_) / contraste(image_contraste, zone_)


def gain_radial(image_reference, image_contraste, r_in=0.0, r_out=config.DEMI_CHAMP_LD, pas=0.25, cote=0):
    """Le profil du gain : `contraste_radial(référence) / contraste_radial(image)`, anneau par anneau.

    Mêmes anneaux que `contraste_radial`. Rend (rayons, gains), deux tableaux (n_anneaux,).
    """
    rayons, reference = contraste_radial(image_reference, r_in, r_out, pas, cote)
    _, contrastes = contraste_radial(image_contraste, r_in, r_out, pas, cote)
    return rayons, reference / contrastes
