# -*- coding: utf-8 -*-
"""
ZELDA : l'image d'un masque de Zernike, et sa lecture.

La physique est celle de pyzelda (N'Diaye et al. 2013), réécrite ici pour que
Manet puisse la dériver : pyzelda ne sert plus que de référence dans les tests.

    image        E = P·e^{iφ},  E_c = E − (1 − e^{iθ})·TF⁻¹[M·TF(E)],  I = |E_c|²
    reconstruire le développement au second ordre en φ, P = 1 : la formule
                 de `pyzelda.ZernikeSensor.analyze`, à l'identique

Le masque peut être décalé (`config.Decalage`) : on met devant le masque centré
la phase équivalente (`optique.phase_masque_nm`). La lecture, elle, suppose
toujours le masque à sa place : c'est ce que fait le banc.
"""
from functools import lru_cache

import numpy as np

from . import config, optique


def indice_silice(lambda_m):
    """L'indice de la silice fondue (Malitson 1965), comme `pyzelda.ztools.refractive_index`."""
    l2 = (lambda_m * 1e6) ** 2
    return np.sqrt(1 + 0.6961663 * l2 / (l2 - 0.0684043 ** 2)
                   + 0.4079426 * l2 / (l2 - 0.1162414 ** 2)
                   + 0.8974794 * l2 / (l2 - 9.896161 ** 2))


DEPHASAGE = 2 * np.pi * (indice_silice(config.LAMBDA) - 1) * config.ZELDA_PROFONDEUR_M / config.LAMBDA
DIAMETRE_LD = config.ZELDA_DIAMETRE_M / (config.LAMBDA * config.F_RATIO)     # 1,06 λ/D

#----------------------------------------On simule ce que la camera voit en ADU broski gng -------------------------------#




@lru_cache
def _foyer(n=config.N_PUP):
    """La MFT vers le masque et le masque échantillonné : PX_MASQUE px sur son diamètre."""
    pas = DIAMETRE_LD / config.PX_MASQUE    #1,06/128= 0.0083 lambda/D
    n_bloc = config.PX_MASQUE + 2   # taille du masque+incertitide de 2pix
    matrice = optique._figer(optique.matrice_mft(n, n_bloc, pas)) #le lien masque zelda --> pupille
    masque = optique._figer(optique.disque(n_bloc, config.PX_MASQUE)) ## matrice booleene indiquant TRue si on est dans la pastille qui depahse
    return matrice, masque, pas


def _filtre(champ, n):
    """TF⁻¹[M·TF(champ)] : la partie du champ que le masque déphase."""
    matrice, masque, pas = _foyer(n)
    return optique.vers_pupille(masque * optique.vers_foyer(champ, matrice), matrice, pas)

########### LA FNC FINALEEEEEEEE YEAH#######
def image(phase_nm, decalage=config.Decalage(), n=config.N_PUP):
    """L'image ZELDA normalisée par la pupille claire, I/|P|², nulle hors de la pupille."""
    pup = optique.pupille(n)
    champ = pup * np.exp(1j * optique.phase_rad(phase_nm + optique.phase_masque_nm(decalage, n)))
    champ_sortie = champ - (1 - np.exp(1j * DEPHASAGE)) * _filtre(champ, n) ### C'est babinet
    return np.abs(champ_sortie) ** 2 * pup




#----------------------------Calcul de l'OPD--------------------------------------------------#
@lru_cache
def onde_reference(n=config.N_PUP):
    """ On calcule b = TF⁻¹[M·TF(P)] : l'onde de référence du masque centré, partie réelle."""
    pup = optique.pupille(n)
    return optique._figer(np.real(_filtre(pup.astype(complex), n)) * pup)


def onde_reference_reelle(phase_nm, n=config.N_PUP):
    """b = Re TF⁻¹[M·TF(P·e^{iφ})] : l'onde de référence calculée avec une phase, pas sur un plat."""
    pup = optique.pupille(n)
    return np.real(_filtre(pup * np.exp(1j * optique.phase_rad(phase_nm)), n)) * pup


def reconstruire(image_normalisee, n=config.N_PUP, b=None, iterations=0):
    """L'OPD lue, en nm, moyenne nulle sur la pupille : la formule de pyzelda (P = 1).

        δ = sin²θ − 2(b − 1)(1 − cos θ)² − (1 − I)(1 − cos θ)/b
        φ = (−sin θ + √δ) / (1 − cos θ)

    Les δ négatifs (hors du domaine de la formule) sont mis à zéro, comme pyzelda.
    b : l'onde de référence, `onde_reference` (plat, comme le banc) si None.
    iterations : relectures avec b recalculé sur la phase lue (`onde_reference_reelle`) ;
    0 fait la lecture du banc.
    """
    if b is None:
        b = onde_reference(n)
    lue = _formule(image_normalisee, b, n)
    for _ in range(iterations):
        lue = _formule(image_normalisee, onde_reference_reelle(lue, n), n)
    return lue


def _formule(image_normalisee, b, n):
    """La formule de pyzelda pour une onde de référence b donnée."""
    dedans = optique.pupille(n) > 0
    b = b[dedans]
    intensite = image_normalisee[dedans]
    c, s = np.cos(DEPHASAGE), np.sin(DEPHASAGE)
    delta = s ** 2 - 2 * (b - 1) * (1 - c) ** 2 - (1 - intensite) * (1 - c) / b
    phase = (-s + np.sqrt(np.clip(delta, 0.0, None))) / (1 - c)
    opd = phase * config.LAMBDA / (2 * np.pi) * 1e9
    sortie = np.zeros(image_normalisee.shape)
    sortie[dedans] = opd - opd.mean()
    return sortie


def lire(phase_nm, decalage=config.Decalage(), n=config.N_PUP, b=None, iterations=0):
    """Ce que ZELDA lit d'une phase : l'image, puis sa reconstruction.

    b : l'onde de référence, `onde_reference` (plat, comme le banc) si None ;
    `onde_reference_reelle(phase_nm)` donne la lecture avec le b exact (simulation seulement).
    """
    return reconstruire(image(phase_nm, decalage, n), n, b=b, iterations=iterations)
