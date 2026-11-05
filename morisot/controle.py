# -*- coding: utf-8 -*-
"""
Le contrôle : l'état du banc, la boucle NCPA de ZELDA, l'EFC et l'iEFC.

Chaque étape prend (banc, état) et rend un nouvel état : l'ordre se choisit
dans le script.

    etat = depart(commande_banc)
    etat = ncpa_compensation(banc, etat)
    etat = ajouter_residu(etat, rms_nm=4)
    G = jacobien_efc(banc, etat, modes, zone)
    etat = creuser_efc(banc, etat, G, modes, zone)

Le SLM applique l'opposé de sa commande : le banc corrige par best_flat += opd
(MITHICstart, ncpa.py). La phase vue vaut donc statique − commande.
"""
from dataclasses import dataclass, replace #pour conserver et creer des nouveau etats sans ecraser
from pathlib import Path

import numpy as np

from morisot import config, coronographe, figures, optique, zelda


@dataclass(frozen=True)
class Etat:
    """Ce que le banc a dans un état donné. Cartes en nm, sur la grille de la pupille.

    statique : l'aberration commune, vue par ZELDA et par le coronographe
    commande : ce que le SLM écrit ; il en applique l'opposé
    science  : l'aberration que seul le coronographe voit (nulle par défaut)
    journal  : une ligne par mesure, de la première étape à la dernière
    """
    statique: np.ndarray
    commande: np.ndarray
    science: np.ndarray
    journal: tuple = ()

    def phase_zelda(self):
        return self.statique - self.commande

    def phase_science(self):
        return self.statique - self.commande + self.science


def commande_banc():
    """La commande du SLM du banc en boucle fermée : command_final de loop_direct_02, 04/09."""
    return np.load(config.DONNEES / 'commande_banc.npy')


def depart(commande_banc, echelle=1.0):
    """Le SLM à plat devant l'aberration que la commande du banc annulait.

    commande_banc : la commande du banc en boucle fermée, sur ses 418 px
    (donnees/commande_banc.npy, command_final du 04/09). À convergence,
    statique − commande ≈ 0 : l'aberration vaut la commande. Son tip/tilt est du
    pointage, on le retire. `echelle` = 0,8 si l'on croit au gain du vrai SLM.
    """
    statique = optique.sans_piston_tip_tilt(optique.depuis_banc(commande_banc)) * echelle
    zero = np.zeros_like(statique)
    return Etat(statique=statique, commande=zero, science=zero.copy())


def ajouter_residu(etat, rms_nm=4.0, voie='commune', exposant=2.0, f_min=1.0, f_max=None, graine=0):
    """Ajoute un écran de `rms_nm` (DSP en f^−exposant) à la voie commune ou à la seule voie science."""
    carte = optique.ecran(rms_nm, exposant, f_min, f_max, graine)
    if voie == 'commune':
        return replace(etat, statique=etat.statique + carte)
    if voie == 'science':
        return replace(etat, science=etat.science + carte)
    raise ValueError("voie : 'commune' ou 'science'")


# ------------------------------------------------------------ la boucle NCPA
def ncpa_compensation(banc, etat, n_iter=config.NCPA_N_ITER, gain=config.NCPA_GAIN,
                      passe_bas=config.NCPA_PASSE_BAS, sans_tip_tilt=True, sans_focus=False, bruit=None,
                      iterations_b=0, plot=False):
    """La boucle ZELDA du banc, `n_iter` tours (MITHICstart, procédure directe, étape 3).

        opd lue → piston/tip/tilt retirés → passe-bas de Hann → commande += gain·opd

    sans_focus : True retire aussi le défocus (Z4) de la lecture : la boucle ne le corrige pas.
    bruit : None, ou une fonction image → image bruitée (par exemple
    `lambda img: cameras.zelda(img, rng)`). iterations_b : relectures de chaque
    image avec l'onde de référence recalculée sur la phase lue (0 : le banc).
    plot : True enregistre la fermeture de boucle en gif dans resultats/ncpa_compensation.gif,
    un chemin l'enregistre là ; une image par tour, les paramètres inscrits dans la figure.
    Il trace aussi le bilan avant/après (phase, caméra ZELDA, caméra science, décalages des
    masques) et la convergence, enregistrés à côté en .png et _convergence.png, laissés ouverts.
    Le journal reçoit n_iter + 1 lignes : la ligne k est l'état après k corrections.
    """
    #on prepare
    commande = etat.commande #on part de la commande actuelle du SLM
    journal = list(etat.journal) #on copie l'historique en liste
    trames = [] #pour tracer , utile seulement pour plot=true
    #on boucle
    for tour in range(n_iter + 1):
        phase = etat.statique - commande #en pratique pn part generalement de phase=l'etat statique n
        img = zelda.image(phase, banc.zelda)
        if bruit is not None: #on bruite 
            img = bruit(img)
        lue = zelda.reconstruire(img, iterations=iterations_b)
        retires = (1, 2, 3) * sans_tip_tilt + (4,) * sans_focus
        if retires:
            lue = optique.sans_modes(lue, retires)
        journal.append(dict(etape='ncpa', tour=tour, rms_vraie=optique.rms(phase),
                            rms_lue=optique.rms(lue)))
        if plot:
            trames.append((phase, img, lue, commande))
        if tour == n_iter:
            break
        commande = commande + gain * optique.passe_bas(lue, passe_bas) #commande finale yesss
    if plot:
        parametres = (f"n_iter = {n_iter}   gain = {gain}   passe_bas = {passe_bas} c/p   "
                      f"sans_tip_tilt = {sans_tip_tilt}   sans_focus = {sans_focus}   bruit ={'oui' if bruit else 'non'}   "
                      f"iterations_b = {iterations_b}\n"
                      f"ZELDA : {banc.zelda}   FPM : {banc.fpm}   Lyot : {banc.lyot}")
        chemin = config.RESULTATS / 'ncpa_compensation.gif' if plot is True else Path(plot)
        figures.gif_ncpa(trames, journal[-(n_iter + 1):], parametres, chemin)
        figures.bilan_ncpa(banc, etat, replace(etat, commande=commande), trames, journal[-(n_iter + 1):],
                           parametres, chemin.with_suffix('.png'))
    return replace(etat, commande=commande, journal=tuple(journal))


# --------------------------------------------------------------------- l'EFC
def modes_controle(r_in=config.R_IN, r_out=config.R_OUT, marge=config.MARGE_MODES):
    """Les modes de Fourier qui posent leurs speckles dans l'anneau, élargi de `marge` c/p."""
    #IWA= 3lambda/d
    #OWA= aucun en pratique
    #la margevient du fit qu'un specle est etaleé et donc si un sepckle est proche de rout, il sera pas corrigee2 complement
    modes, _ = optique.modes_fourier(r_in - marge, r_out + marge)
    return modes #modes est une liste


def _ajouter(commande, coefficients, modes):
    #on prends la commande actuelle du SLM (par ex bestflat) eton ajt les modes de fourier avec leur coef
    """La commande plus Σ coefficients_k · mode_k."""
    return commande + np.tensordot(coefficients, modes, axes=1)



def _mesure_champ(champ, zone_):
    """On prend E, matrice compexe et on retourne une liste tq [Re E ; Im E] sur la zone : ce que l'EFC annule."""
    return np.concatenate([champ[zone_].real, champ[zone_].imag])

############################################# LA MATRICE IMPORTANTE #########################################


def jacobien_efc(banc, etat, modes, zone_, pas_nm=2.0):
    """Le jacobien [Re ; Im] du champ dans la zone, par mode ajouté à la commande.

    Différence centrée de ±pas_nm : (2·n_zone, n_modes). Manet le calcule exact,
    par différentiation automatique.

    Simple matrice d'interaction entre l'espace du DM et l'espace electrique en plan focal
    #colonne
    #ligne:


    # nouveauté par rapport a GHOST on peut estimer le jacobien en simulant des NCPA que ZELDA ne voit pas
    """
    phase = etat.phase_science()
    
    colonnes = []
    for mode in modes:
        ################## ATTTENTION C"EST BIEN LE CHAMP PAS L'IMAGE ################
        plus = coronographe.champ(phase - pas_nm * mode, banc)      # commande + pas
        moins = coronographe.champ(phase + pas_nm * mode, banc)
        colonnes.append(_mesure_champ((plus - moins) / (2 * pas_nm), zone_))  # matrice ineraction tout cutie mimi 
    return np.stack(colonnes, axis=1)


################################################################################################################

# Petit rappel. 
# Dans notre cas, pourquoi faison nous du Pair wise Probing pour avoir la matrice EFC ou la matrice iEFC?
#matrice EFC: lien entre modes dee fourier et champ en plan focal (sous reserve de connaitre le cham fait par les probes)
    #c'est une derive du champs par la les modes
#Matrice iEFC: lien entre les commandes appliquées par le SLM et l'intensite meseuré
    #c'est une dervee dseconde de l'intensité par les sondes et les modes
    #l'intensite quelle quantite annuler -> equivalent de connaitre l'amplitude du champ
    #les sondes par dessus :dans quel sens annuler. --> reconstruction de la phas et l'amplitude e du champ

#Tout deux passent par une mesure de difference d'intensité
#Si nou sfaisions comme en OA par exemple, avec une simple matrice d'interaction ou on met un modes et on regarde sa reponse en plan pupille.
#On aurait aucun moyen de remonter aux champ electirque aue l'on veut annuler
#Pour cela, on utilise de l'interference en creant deux ondes dephasé celle du champ E avec celle du champ faite par le probe
#Donc on a I(p)= |E+Ep|**2=|E|**2 + |Ep|**2 + 2Re(E*Ep) seulement on veut connaitre seulement le champ Ep pas le terme d'intensité
#On fait la difference avec le meme probe en negatfi, on obtient:
#I(+p)-I(-p)= 4 Re (EpE)

def reconstructeur(matrice, alpha):
    #on inverse G
    """Tikhonov rapporté à la valeur propre moyenne : (GᵀG + α·tr(GᵀG)/n·I)⁻¹Gᵀ."""
    gtg = matrice.T @ matrice
    regul = alpha * np.trace(gtg) / gtg.shape[0]
    return np.linalg.solve(gtg + regul * np.eye(gtg.shape[0]), matrice.T)

################################### la boucle pour avoir le DH ###################################
def _creuser(banc, etat, mesurer, inverse, modes, zone_, gain, n_iter, etape, plot=False, parametres=''):
    """La boucle de plan focal commune à l'EFC et à l'iEFC : Δa = −R·m(commande).
    Cette fonction retorune le nouvel etat"""
    
    commande = etat.commande #on est au best flat du SLM
    journal = list(etat.journal)
    trames = []
    for tour in range(n_iter + 1):
        img = coronographe.image(etat.statique - commande + etat.science, banc) #on regrde ce que voit la science camera 
        journal.append(dict(etape=etape, tour=tour, contraste=coronographe.contraste(img, zone_))) #et on l'enregistre ds le journal de bord
        if plot:  #on garde de quoi fare le GIF
            trames.append((img, etat.statique - commande + etat.science, commande - etat.commande))
        if tour == n_iter:
            break
        ######################################## LA COMMANDE #####################################################
        commande = _ajouter(commande, -gain * (inverse @ mesurer(commande)), modes) # je raprlle que le _ajouter est pour convertir en commande SLM
        ##########################################################################################################
    if plot:
        parametres = (f"{etape.upper()}   n_iter = {n_iter}   gain = {gain}   {parametres}   "
                      f"{len(modes)} modes\n"
                      f"ZELDA : {banc.zelda}   FPM : {banc.fpm}   Lyot : {banc.lyot}")
        chemin = config.RESULTATS / f'creuser_{etape}.gif' if plot is True else plot
        figures.gif_dark_hole(trames, journal[-(n_iter + 1):], zone_, parametres, chemin)
    return replace(etat, commande=commande, journal=tuple(journal))



def ideal_efc(banc, etat, matrice, modes, zone_, alpha=1e-2, gain=config.GAIN_EFC, n_iter=10, plot=False):

    """L'EFC idéale : le vrai champ complexe dans la zone, lu dans la simulation, et le jacobien.

    Aucun banc ne mesure ce champ : c'est la limite de l'EFC, sans erreur d'estimation.
    `real_efc` l'estime par pair-wise probing, comme au banc.
    plot : True enregistre les tours en gif dans resultats/creuser_efc.gif, un chemin l'enregistre là.
    """
    inverse = reconstructeur(matrice, alpha)

    def mesurer(commande):
        return _mesure_champ(coronographe.champ(etat.statique - commande + etat.science, banc), zone_)
    return _creuser(banc, etat, mesurer, inverse, modes, zone_, gain, n_iter, 'efc',
                    plot, f"alpha = {alpha}")


creuser_efc = ideal_efc     # l'ancien nom, gardé pour les scénarios, Manet et les comparaisons





# -------------------------------------------------------------------- l'iEFC

def sondes(r_in=config.R_IN, r_out=config.R_OUT, cote=+1, rms_nm=2.0):
    """Deux sondes sinc·sinc·cos et sinc·sinc·sin qui éclairent le demi-anneau.

    Axel Potier, utilise des sondes dans sa these ou c'est simplement des pusch pull d'actionneurs centraux, tF d'un dirac, c'est cst.
    Comme on a un SLM, bbbbccccp d'actionneur, je trouce ca plus adapte2 de faire un sinc! plus smooth

    sinc(w·x) a pour transformée une fente de w λ/D ; la porteuse la pose au
    milieu du demi-anneau. Chaque sonde vaut `rms_nm` sur la pupille.

    ##
    """
    x, y = optique.coordonnees()
    pup = optique.pupille()
    centre = (r_in + r_out) / 2.0
    enveloppe = np.sinc((r_out - r_in + 2) * x) * np.sinc((2 * r_out + 2) * y)
    sorties = []
    for porteuse in (np.cos, np.sin):
        carte = enveloppe * porteuse(2 * np.pi * cote * centre * x) * pup
        sorties.append(carte * rms_nm / carte[pup > 0].std())
    return np.array(sorties)

############################### Partie subtile ######################################################

def _mesure_iefc(banc, etat, commande, sondes_, zone_):
    """[I(c + s_p) − I(c − s_p)] sur la zone, pour chaque sonde : rien que des images."""
    phase = etat.statique - commande + etat.science
    return np.concatenate([coronographe.image(phase - sonde, banc)[zone_]
                           - coronographe.image(phase + sonde, banc)[zone_] for sonde in sondes_])


def matrice_iefc(banc, etat, modes, zone_, sondes_, pas_nm=2.0):
    """La matrice de l'iEFC, mesurée par images : quatre poses par mode et par sonde."""
    colonnes = []
    for mode in modes:
        plus = _mesure_iefc(banc, etat, etat.commande + pas_nm * mode, sondes_, zone_)
        moins = _mesure_iefc(banc, etat, etat.commande - pas_nm * mode, sondes_, zone_)
        colonnes.append((plus - moins) / (2 * pas_nm))
    return np.stack(colonnes, axis=1)

######################################################################################################
def creuser_iefc(banc, etat, matrice, modes, zone_, sondes_, alpha=1e-2, gain=config.GAIN_EFC,
                 n_iter=10, plot=False):
    """L'iEFC : la même boucle, mesurée par sondes au lieu du champ du modèle.

    plot : True enregistre les tours en gif dans resultats/creuser_iefc.gif, un chemin l'enregistre là.
    """
    inverse = reconstructeur(matrice, alpha)

    def mesurer(commande):
        return _mesure_iefc(banc, etat, commande, sondes_, zone_)
    rms_sondes = optique.rms(sondes_[0])
    return _creuser(banc, etat, mesurer, inverse, modes, zone_, gain, n_iter, 'iefc',
                    plot, f"alpha = {alpha}   {len(sondes_)} sondes de {rms_sondes:.1f} nm rms")


# ------------------------------------------------ l'EFC réaliste : pair-wise probing
def sondes_centrales(cote_px=16, amplitude_nm=50.0, ecart_px=12):
    """Trois sondes push-pull : un carré de pixels du SLM poussé de `amplitude_nm`, au centre,
    puis décalé de `ecart_px` en x, puis en y (grille de Morisot ; 1 px = N_BANC/N_PUP px du SLM).

    Un carré de c px éclaire le foyer jusqu'à N_PUP/c λ/D : 16 px couvrent tout le dark hole,
    avec un contraste d'environ 1e-5 à 50 nm. Un pixel seul ne donne que 2e-10, invisible.
    Décaler la sonde tourne la phase de son champ de 2π(u·dx + v·dy) : avec deux sondes
    seulement, une ligne du dark hole n'a pas de diversité, d'où la troisième.
    """
    n = config.N_PUP
    debut = n // 2 - cote_px // 2
    sorties = []
    for dx, dy in ((0, 0), (ecart_px, 0), (0, ecart_px)):
        carte = np.zeros((n, n))
        carte[debut + dy:debut + dy + cote_px, debut + dx:debut + dx + cote_px] = amplitude_nm
        sorties.append(carte * optique.pupille())
    return np.array(sorties)


def estimer_champ_pwp(banc, etat, commande, sondes_, zone_, banc_modele=config.Banc(), bruit=None,
                      regul_pwp=1e-2):
    """Le champ complexe dans la zone, estimé par pair-wise probing (Give'on et al. 2011).

    Pour chaque sonde s, deux images avec la commande ± s : I(c + s) − I(c − s) = 4·Re(E*·ΔE_s),
    où ΔE_s, le champ que crée la sonde, vient du modèle. Avec au moins deux sondes, chaque
    pixel donne Re E et Im E par moindres carrés.

    Le modèle est celui qu'on aurait au banc : il ne connaît pas les aberrations et suppose
    le faisceau plat (la commande les compense), et ses masques sont à `banc_modele`
    (nominaux par défaut), pas forcément là où ils sont vraiment (`banc`).
    bruit : None, ou une fonction image → image bruitée (par exemple
    `lambda img: cameras.science(img, rng)`), appliquée à chaque pose.
    regul_pwp : la régularisation par pixel, rapportée à l'énergie médiane des sondes.
    sondes_ : `sondes_centrales()` (push-pull au centre du SLM) ou `sondes()`.
    Rend (n_zone,) complexe.
    """
    phase = etat.statique - commande + etat.science

    def poser(phase_pose):
        img = coronographe.image(phase_pose, banc)
        return (bruit(img) if bruit is not None else img)[zone_]

    # commande + s ↔ phase − s
    differences = np.stack([poser(phase - s) - poser(phase + s) for s in sondes_], axis=1)
    modele = np.zeros_like(phase)
    champs_sondes = np.stack([(coronographe.champ(modele - s, banc_modele)
                               - coronographe.champ(modele + s, banc_modele))[zone_] / 2
                              for s in sondes_], axis=1)                   # (n_zone, n_sondes)
    a = np.stack([champs_sondes.real, champs_sondes.imag], axis=-1)       # (n_zone, n_sondes, 2)
    ata = np.einsum('kpi,kpj->kij', a, a)
    atb = np.einsum('kpi,kp->ki', a, differences / 4)
    # là où une sonde s'annule (un zéro du sinc), le système est singulier : on régularise,
    # ce qui ramène l'estimation vers 0 dans ces pixels au lieu de la faire exploser
    regul = regul_pwp * np.median(np.trace(ata, axis1=1, axis2=2)) / 2
    re_im = np.linalg.solve(ata + regul * np.eye(2), atb[..., None])[..., 0]
    return re_im[:, 0] + 1j * re_im[:, 1]


def real_efc(banc, etat, matrice, modes, zone_, sondes_, banc_modele=config.Banc(), alpha=1e-2,
             gain=config.GAIN_EFC, n_iter=10, bruit=None, plot=False):
    """L'EFC comme au banc : le champ estimé par pair-wise probing, le jacobien du modèle.

    matrice : le jacobien, à calculer sur le modèle (par exemple `jacobien_efc(banc_modele, etat_modele, ...)`).
    sondes_ : les sondes du pair-wise probing, par exemple `sondes_centrales()`.
    banc_modele : où le modèle croit les masques (nominaux par défaut) ; banc : où ils sont.
    Chaque tour coûte 2 poses par sonde. Comparer à `ideal_efc` (le vrai champ) et à
    `creuser_iefc` (aucun modèle).
    plot : True enregistre les tours en gif dans resultats/creuser_real_efc.gif, un chemin l'enregistre là.
    """
    inverse = reconstructeur(matrice, alpha)

    def mesurer(commande):
        champ = estimer_champ_pwp(banc, etat, commande, sondes_, zone_, banc_modele, bruit)
        return np.concatenate([champ.real, champ.imag])
    return _creuser(banc, etat, mesurer, inverse, modes, zone_, gain, n_iter, 'real_efc', plot,
                    f"alpha = {alpha}   pair-wise probing : {len(sondes_)} sondes   "
                    f"bruit = {'oui' if bruit else 'non'}")
