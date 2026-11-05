# -*- coding: utf-8 -*-
"""
MORISOT: MITHiC Optical Replica In Software for Optimization and Testing

Tout est idéal : SLM exact, aucune misregistration, masques à leur place sauf
si on les décale. La seule donnée importée du banc est sa commande en boucle
fermée (donnees/commande_banc.npy), qui fixe l'aberration statique.

    config        les constantes du banc, Decalage, DecalageLyot, Banc
    optique       pupille, MFT, masques, passe-bas, modes, écrans
    zelda         l'image ZELDA et sa lecture (formule de pyzelda)
    coronographe  le Lyot du banc, l'image en contraste, le demi-anneau
    cameras       bruit de photons et de lecture des deux caméras
    controle      l'état, la boucle NCPA, l'EFC, l'iEFC
    figures       les gifs et les bilans des boucles (plot=True)
    turbulence    les écrans d'OOPAO et la pose longue (à importer à part :
                  `from morisot import turbulence`, il demande OOPAO)

Manet est le même jumeau en PyTorch, différentiable, avec les mêmes fonctions.
MANET: MITHiC Autodifferentiable Numerical Emulator with Torch
"""
from . import config, optique, zelda, coronographe, cameras, figures, controle  # noqa: F401
from .config import Banc, Decalage, DecalageLyot  # noqa: F401
from .controle import Etat  # noqa: F401
