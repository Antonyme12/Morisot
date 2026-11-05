# -*- coding: utf-8 -*-
"""
Les figures des boucles : le gif et le bilan de la boucle NCPA, le gif du dark hole.

Rien de physique ici : chaque fonction reçoit ce que la boucle a enregistré
(`trames`, `journal`) et le dessine. matplotlib n'est importé qu'au moment de
tracer, pour que le reste de Morisot n'en dépende pas.
"""
import numpy as np

from morisot import config, coronographe, optique


# ------------------------------------------------------------ la boucle NCPA
def bilan_ncpa(banc, avant, apres, trames, journal, parametres, chemin):
    """La figure avant/après de la boucle NCPA : une ligne par état, la convergence à droite."""
    import matplotlib.pyplot as plt

    dedans = optique.pupille() > 0
    zone_ = coronographe.zone()
    borne = np.percentile(np.abs(avant.phase_zelda()[dedans]), 99)
    champ_ld = [-config.DEMI_CHAMP_LD, config.DEMI_CHAMP_LD] * 2
    z, f, l = banc.zelda, banc.fpm, banc.lyot
    decalage_zelda = f"ZELDA : dx = {z.dx_um:g} µm, dy = {z.dy_um:g} µm, dz = {z.dz_mm:g} mm"
    decalage_science = (f"FPM : dx = {f.dx_um:g} µm, dy = {f.dy_um:g} µm, dz = {f.dz_mm:g} mm\n"
                        f"Lyot : dx = {l.dx:g} D, dy = {l.dy:g} D")

    fig = plt.figure(figsize=(15, 9))
    grille = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.45)
    for ligne, (nom, etat, img) in enumerate([("AVANT", avant, trames[0][1]),
                                              ("APRÈS", apres, trames[-1][1])]):
        phase = etat.phase_zelda()
        science = coronographe.image(etat.phase_science(), banc)
        axes = [fig.add_subplot(grille[ligne, k]) for k in range(3)]
        vues = [axes[0].imshow(np.where(dedans, phase, np.nan), cmap='RdBu_r', vmin=-borne, vmax=borne),
                axes[1].imshow(np.where(dedans, img, np.nan), vmin=0, vmax=2),
                axes[2].imshow(np.log10(np.clip(science, 1e-12, None)), cmap='inferno', vmin=-9, vmax=-3,
                               extent=champ_ld)]
        titres = [f"phase vue par ZELDA\n{optique.rms(phase):.2f} nm rms",
                  f"caméra ZELDA\n{decalage_zelda}",
                  f"caméra science (log10), contraste 3–8 λ/D = "
                  f"{coronographe.contraste(science, zone_):.1e}\n{decalage_science}"]
        etiquettes = ["nm", "I / pupille claire", "log10(contraste)"]
        for axe, vue, titre, etiquette in zip(axes, vues, titres, etiquettes):
            axe.set_title(titre, fontsize=9)
            fig.colorbar(vue, ax=axe, fraction=0.046, label=etiquette)
        for axe in axes[:2]:
            axe.axis('off')
        axes[2].set_xlabel('u (λ/D)', fontsize=8)
        axes[2].set_ylabel('v (λ/D)', fontsize=8)
        axes[0].text(-0.25, 0.5, nom, transform=axes[0].transAxes, rotation=90, va='center',
                     fontsize=15, fontweight='bold')

    fig.suptitle("Boucle NCPA : avant / après\n" + parametres.split('\n')[0], fontsize=10)
    fig.savefig(chemin, dpi=80, bbox_inches='tight')

    fig_c, courbe = plt.subplots(figsize=(8, 5))
    tours = [l['tour'] for l in journal]
    courbe.semilogy(tours, [l['rms_vraie'] for l in journal], 'o-', color='tab:red', label='rms vraie')
    courbe.semilogy(tours, [l['rms_lue'] for l in journal], 's--', color='tab:blue', label='rms lue par ZELDA')
    courbe.set_xlabel('tour')
    courbe.set_ylabel('nm rms')
    courbe.grid(alpha=0.3, which='both')
    courbe.legend()
    courbe.set_title("Convergence de la boucle NCPA\n" + parametres.replace('   ', '  '), fontsize=8)
    fig_c.savefig(chemin.with_name(chemin.stem + '_convergence.png'), dpi=80, bbox_inches='tight')
    return fig, fig_c


def gif_ncpa(trames, journal, parametres, chemin, fps=2):
    """Le gif de la boucle NCPA : phase vraie, image ZELDA, phase lue, commande, convergence."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    dedans = optique.pupille() > 0
    tours = [ligne['tour'] for ligne in journal]
    vraies = [ligne['rms_vraie'] for ligne in journal]
    lues = [ligne['rms_lue'] for ligne in journal]
    borne = np.percentile(np.abs(trames[0][0][dedans]), 99)

    def masquer(carte):
        return np.where(dedans, carte, np.nan)

    fig = plt.figure(figsize=(16, 7.5))
    grille = fig.add_gridspec(2, 4, height_ratios=[1, 0.8], hspace=0.35)
    cartes = [fig.add_subplot(grille[0, k]) for k in range(4)]
    courbe = fig.add_subplot(grille[1, :])
    images = [cartes[0].imshow(masquer(trames[0][0]), cmap='RdBu_r', vmin=-borne, vmax=borne),
              cartes[1].imshow(masquer(trames[0][1]), vmin=0, vmax=2),
              cartes[2].imshow(masquer(trames[0][2]), cmap='RdBu_r', vmin=-borne, vmax=borne),
              cartes[3].imshow(masquer(trames[0][3]), cmap='RdBu_r', vmin=-borne, vmax=borne)]
    for axe, image in zip(cartes, images):
        axe.axis('off')
        fig.colorbar(image, ax=axe, fraction=0.046)
    courbe.semilogy(tours, vraies, color='0.8')
    courbe.semilogy(tours, lues, color='0.8')
    ligne_vraie, = courbe.semilogy([], [], 'o-', color='tab:red', label='rms vraie')
    ligne_lue, = courbe.semilogy([], [], 's--', color='tab:blue', label='rms lue par ZELDA')
    courbe.set_xlabel('tour')
    courbe.set_ylabel('nm rms')
    courbe.grid(alpha=0.3, which='both')
    courbe.legend()
    fig.suptitle(parametres, fontsize=10)

    def tracer(k):
        phase, img, lue, commande = trames[k]
        for image, carte in zip(images, (phase, img, lue, commande)):
            image.set_data(masquer(carte))
        cartes[0].set_title(f'phase vraie : {vraies[k]:.2f} nm rms')
        cartes[1].set_title(f'image ZELDA, tour {k}/{len(trames) - 1}')
        cartes[2].set_title(f'phase lue : {lues[k]:.2f} nm rms')
        cartes[3].set_title(f'commande SLM : {optique.rms(commande):.1f} nm rms')
        ligne_vraie.set_data(tours[:k + 1], vraies[:k + 1])
        ligne_lue.set_data(tours[:k + 1], lues[:k + 1])
        return images + [ligne_vraie, ligne_lue]

    FuncAnimation(fig, tracer, frames=len(trames)).save(chemin, writer=PillowWriter(fps=fps), dpi=70)
    plt.close(fig)


# ------------------------------------------------------------- le dark hole
def gif_dark_hole(trames, journal, zone_, parametres, chemin, fps=2):
    """Le gif d'une boucle de plan focal : image coronographique, phase, ce que la boucle a ajouté, contraste."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    dedans = optique.pupille() > 0
    tours = [ligne['tour'] for ligne in journal]
    contrastes = [ligne['contraste'] for ligne in journal]
    demi = config.DEMI_CHAMP_LD
    borne_phase = np.percentile(np.abs(trames[0][1][dedans]), 99)
    borne_ajout = max(np.percentile(np.abs(trames[-1][2][dedans]), 99), 1e-3)

    def masquer(carte):
        return np.where(dedans, carte, np.nan)

    fig = plt.figure(figsize=(15, 7.5))
    grille = fig.add_gridspec(2, 3, height_ratios=[1, 0.8], hspace=0.35)
    cartes = [fig.add_subplot(grille[0, k]) for k in range(3)]
    courbe = fig.add_subplot(grille[1, :])
    images = [cartes[0].imshow(np.log10(trames[0][0] + 1e-12), cmap='inferno', vmin=-9, vmax=-3,
                               extent=[-demi, demi, demi, -demi]),
              cartes[1].imshow(masquer(trames[0][1]), cmap='RdBu_r', vmin=-borne_phase, vmax=borne_phase),
              cartes[2].imshow(masquer(trames[0][2]), cmap='RdBu_r', vmin=-borne_ajout, vmax=borne_ajout)]
    cartes[0].contour(np.linspace(-demi, demi, zone_.shape[1]), np.linspace(-demi, demi, zone_.shape[0]),
                      zone_, levels=[0.5], colors='white', linewidths=0.8)
    cartes[0].set_xlabel('λ/D')
    for axe in cartes[1:]:
        axe.axis('off')
    for axe, image, nom in zip(cartes, images, ('log₁₀ contraste', 'nm', 'nm')):
        fig.colorbar(image, ax=axe, fraction=0.046, label=nom)
    courbe.semilogy(tours, contrastes, color='0.8')
    ligne, = courbe.semilogy([], [], 'o-', color='tab:purple', label='contraste moyen dans la zone')
    courbe.set_xlabel('tour')
    courbe.set_ylabel('contraste')
    courbe.grid(alpha=0.3, which='both')
    courbe.legend()
    fig.suptitle(parametres, fontsize=10)

    def tracer(k):
        img, phase, ajout = trames[k]
        images[0].set_data(np.log10(img + 1e-12))
        images[1].set_data(masquer(phase))
        images[2].set_data(masquer(ajout))
        cartes[0].set_title(f'image, tour {k}/{len(trames) - 1} : {contrastes[k]:.2e}')
        cartes[1].set_title(f'phase vue par le coronographe : {optique.rms(phase):.2f} nm rms')
        cartes[2].set_title(f'ajouté par la boucle : {optique.rms(ajout):.2f} nm rms')
        ligne.set_data(tours[:k + 1], contrastes[:k + 1])
        return images + [ligne]

    FuncAnimation(fig, tracer, frames=len(trames)).save(chemin, writer=PillowWriter(fps=fps), dpi=70)
    plt.close(fig)
