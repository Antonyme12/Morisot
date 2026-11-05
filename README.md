# Morisot

**MORISOT** : *MITHiC Optical Replica In Software for Optimization and Testing*.

Le jumeau numérique du banc MITHiC (LAM) parfait : un SLM corrige, un ZELDA mesure la phase,
un coronographe de Lyot (masque de 5,38 λ/D, Lyot de 0,77 D) éteint l'étoile. On y ferme la
boucle NCPA de ZELDA, puis on creuse un dark hole par EFC, iEFC ou pair-wise probing.

Le code est en numpy. Ce sont des fonctions indépendantes qu'on enchaîne dans un script, avec un
état figé qu'on fait passer d'étape en étape. Tout est idéal : SLM exact, aucune misregistration,
masques à leur place sauf si on les décale. La seule donnée importée du banc est sa commande en
boucle fermée, qui fixe l'aberration statique.

## Installer

```bash
git clone https://github.com/Antonyme12/Morisot.git
cd Morisot
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

`-e` installe le dossier lui-même, sans copie : on modifie le code, Python voit la modification.
La commande du banc (`donnees/`) est lue à côté du code, d'où cette installation plutôt qu'un
`pip install .`.

Le module `turbulence` demande OOPAO, à installer depuis son dépôt GitHub
(github.com/cheritier/OOPAO) ; le reste de Morisot ne dépend pas de lui.

## Simuler

```python
import morisot
from morisot import controle, coronographe

banc = morisot.Banc()                                  # les trois masques à leur place
etat = controle.depart(controle.commande_banc())       # le SLM à plat devant la statique du banc
etat = controle.ncpa_compensation(banc, etat)          # la boucle ZELDA du banc, 15 tours
etat = controle.ajouter_residu(etat, rms_nm=4)         # un résidu de 4 nm rms

zone = coronographe.zone()                             # le demi-anneau 3-8 λ/D
modes = controle.modes_controle()
G = controle.jacobien_efc(banc, etat, modes, zone)
etat = controle.creuser_efc(banc, etat, G, modes, zone)
print(etat.journal[-1]['contraste'])
```

Décaler un masque : `morisot.Banc(zelda=morisot.Decalage(dx_um=4))`, et de même pour `fpm`, ou
`lyot=morisot.DecalageLyot(dx=0.01)`. `plot=True` dans les boucles enregistre un gif dans
`resultats/`.

## Le code

| module | ce qu'il fait |
|---|---|
| `config.py` | les constantes du banc, avec leur source ; `Banc`, `Decalage`, `DecalageLyot` |
| `optique.py` | la pupille, la MFT, les masques, le passe-bas du banc, les modes, les écrans |
| `zelda.py` | l'image ZELDA et sa lecture (la formule de pyzelda) |
| `coronographe.py` | le Lyot du banc, l'image en contraste, le demi-anneau |
| `cameras.py` | le bruit de photons et de lecture des deux caméras |
| `controle.py` | l'état, la boucle NCPA, l'EFC, l'iEFC, le pair-wise probing |
| `figures.py` | les gifs et les bilans des boucles (`plot=True`) |
| `turbulence.py` | les écrans d'OOPAO et la pose longue (demande OOPAO) |

Les conventions : une carte de phase est une OPD en nm sur 256 × 256 px ; x, y sont en unités
de D ; u, v sont en λ/D, et un mode à f c/p pose ses speckles à f λ/D.

## Les données

`donnees/commande_banc.npy` : la commande du SLM de MITHiC en boucle fermée (`command_final` de
`loop_direct_02`, 04/09/2026), sur les 418 px du disque de commande. Ce sont des données du banc :
ne pas rendre ce dépôt public sans l'accord de l'équipe.

## Les fichiers de ce dépôt

| fichier | pourquoi il est là |
|---|---|
| `README.md` | la page d'accueil que GitHub affiche : ce texte |
| `pyproject.toml` | le nom du paquet et ses dépendances ; lu par `pip install -e .` |
| `.gitignore` | ce que git ne doit pas envoyer : les caches de Python, les figures, l'environnement |
| `morisot/` | le code |
| `donnees/` | ce que le code lit du banc |

Pas de licence pour l'instant : sans elle, le code reste à son auteur. À décider avec l'équipe
avant de rendre le dépôt public.

Manet, la même chose en PyTorch et différentiable, n'est pas dans ce dépôt.
