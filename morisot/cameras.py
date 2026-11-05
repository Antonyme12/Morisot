# -*- coding: utf-8 -*-
"""
Les deux caméras : bruit de photons, bruit de lecture, numérisation, saturation.

Une image normalisée entre, une image normalisée et bruitée sort : on peut
mettre une caméra ou l'enlever sans rien changer d'autre. Le générateur
aléatoire est toujours passé en argument.

Les niveaux et les gains sont ceux du banc (séance du 04/09, et la courbe de
transfert de photons de la caméra science). Ils ne sont pas encore calés pour
reproduire la répétabilité de ZELDA mesurée au banc.
"""
import numpy as np

PLAFOND_ADU = 16383             # 14 bits

ZELDA_PUPILLE_ADU = 1725.4      # la pupille claire moyenne du 04/09
ZELDA_GAIN = 0.69               # e-/ADU, déduit du banc (1,6 e- pour 2,32 ADU)
ZELDA_LECTURE_E = 1.6           # e- rms

SCIENCE_PIC_ADU = 14210.0       # le pic de la PSF, dark soustrait (psf_iter025 du 04/09)
SCIENCE_GAIN = 0.792            # e-/ADU, courbe de transfert de photons (la fiche dit 0,95)
SCIENCE_LECTURE_E = 4.93 * 0.792    # e- rms : 4,93 ADU mesurés


def pose(image, niveau_adu, gain_e_adu, lecture_e, rng=None, plafond_adu=PLAFOND_ADU, bruit=True):
    """Une pose, en ADU : l'image normalisée vaut `niveau_adu` là où elle vaut 1.

    bruit=False : ni photons ni lecture, il reste la numérisation et la saturation
    (rng inutile alors).
    """
    electrons = np.clip(image, 0.0, None) * niveau_adu * gain_e_adu
    if bruit:
        electrons = rng.poisson(electrons) + rng.normal(0.0, lecture_e, image.shape)
    return np.minimum(np.round(electrons / gain_e_adu), plafond_adu)


def zelda(image, rng=None, bruit=True):
    """L'image ZELDA normalisée (pupille claire = 1), vue par la caméra ZELDA."""
    return pose(image, ZELDA_PUPILLE_ADU, ZELDA_GAIN, ZELDA_LECTURE_E, rng,
                bruit=bruit) / ZELDA_PUPILLE_ADU


def science(image_contraste, rng=None, temps_relatif=1.0, bruit=True):
    """L'image science en contraste, vue par la caméra ; `temps_relatif` allonge la pose de la PSF."""
    niveau = SCIENCE_PIC_ADU * temps_relatif
    return pose(image_contraste, niveau, SCIENCE_GAIN, SCIENCE_LECTURE_E, rng,
                plafond_adu=np.inf, bruit=bruit) / niveau
