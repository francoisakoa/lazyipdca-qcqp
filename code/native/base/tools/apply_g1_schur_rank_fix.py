#!/usr/bin/env python3
"""Apply the checked G1 normal-rank fix; print the selected source path only."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile

TOP = 'lazyipdca_qcqp_v0.4.28r3p28g1_atomic_graft_source'
SOURCE = 'src/condensed_retraction.cpp'
OLD_SHA = '4fab8baf8baeab56532ccf2fd81c504d3428cb2054021c92bf8c514b0427344b'
NEW_SHA = 'aaa9005f78a06db4b0c130a1a7a8690a308bca8a2b62be852e875f85ca632461'
ANCHOR = '  factor_settings_.kkt_adaptive_regularization = true;\n'
FIX = '''  // The scaled normal operator must detect rank loss up to roundoff. A
  // sign-only native inertia check can accept a spurious O(epsilon) pivot
  // for duplicate equality rows, depending on the BLAS implementation.
  // Keep this guard local: Newton and tangent-curvature settings retain
  // their original inertia policy.
  factor_settings_.native_inertia_relative_zero_tolerance = std::max(
      factor_settings_.native_inertia_relative_zero_tolerance, kTiny);
'''
NOTE = '''# G1-SCHUR-RANK-1 : rang numérique de la rétraction normale

Le test existant condensed_flow_lift_schur peut échouer avec OpenBLAS :
« rank-deficient test did not exercise positive regularization ».
Une égalité exactement dupliquée laisse parfois un pivot d'arrondi négatif.
La classification native par le seul signe accepte alors à tort la matrice
normale singulière et conserve une régularisation nulle.

Le correctif rétablit une tolérance de zéro numérique dans factor_settings_
de CondensedQuadraticRetraction : au moins kTiny = 64 * epsilon_machine,
multiplié par la norme de la matrice par le backend natif. La perte de rang
déclenche ainsi la progression de régularisation déjà prévue dans la rétraction.
Une tolérance utilisateur supérieure est conservée. Cette modification est
locale au facteur normal ; les réglages Newton et du certificat de courbure
tangent restent inchangés. Aucun test n'est désactivé ou assoupli.

Vérification du correctif : les quatre tests existants
test_condensed_flow_lift_schur, test_condensed_quadratic_retraction,
test_g1_lazy_newton et test_g1_curvature_probe passent avec LAPACK système
et avec les routines dsytrf/dsytrs OpenBLAS de SciPy : huit succès.
Le code modifié a été compilé en C++17, -O3, -march=native, avec OpenMP.
La suite complète Conda/LDL/METIS doit être rejouée sur la machine cible.

Le marqueur results/R3P28G1_SOURCE_LAST ne doit être écrit qu'après succès
de la compilation et de CTest. Le correctif n'écrit pas ce marqueur.
La version d'API reste G1, API 4327 ; l'identifiant de maintenance est
G1-SCHUR-RANK-1. Les données et points initiaux de case300 sont inchangés.

Le manifeste courant est mis à jour seulement pour les fichiers de ce
correctif. La source précédente est conservée dans archive_repair/.
Les rapports de réparation de l'archive décrivent l'état antérieur à ce
correctif de code ; leurs mentions « sources inchangées » concernent cette
première réparation uniquement.
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = (path.stat().st_mode & 0o777) if path.exists() else 0o644
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', delete=False) as f:
        tmp = Path(f.name)
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    try:
        tmp.chmod(mode)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('/data/projects/IPDCA_QCQP'))
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    if args.source:
        root = args.source.resolve(strict=True)
    else:
        candidates = [p for p in args.base.glob('r3p28g1_*/' + TOP)
                      if (p / SOURCE).is_file()
                      and (p / 'build_pf_gpu/lazyipdca_qcqp_cli').is_file()]
        if not candidates:
            raise RuntimeError('Aucun build G1 trouvé. Préciser --source /chemin/source_G1.')
        root = max(candidates, key=lambda p: (p / 'build_pf_gpu/lazyipdca_qcqp_cli').stat().st_mtime_ns).resolve()
        print('BUILD_SELECTION=most_recent_cli candidates=' + str(len(candidates)), file=sys.stderr)
    print('SELECTED_SOURCE=' + str(root), file=sys.stderr)
    current = (root / SOURCE).read_bytes()
    current_sha = sha(current)
    if current_sha == OLD_SHA:
        original = current
        if original.count(ANCHOR.encode()) != 1:
            raise RuntimeError('Point de modification ambigu.')
        patched = original.replace(ANCHOR.encode(), (ANCHOR + FIX).encode(), 1)
    elif current_sha == NEW_SHA:
        patched = current
        original = patched.replace(FIX.encode(), b'', 1)
    else:
        raise RuntimeError('Source différente de G1 attendu ; aucune modification : SHA256=' + current_sha)
    if sha(original) != OLD_SHA or sha(patched) != NEW_SHA:
        raise RuntimeError('Contrôle du correctif échoué ; aucune modification.')

    manifest = root / 'RELEASE_MANIFEST_SHA256_G1.txt'
    entries = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(None, 1)
        if name in entries:
            raise RuntimeError('Entrée de manifeste dupliquée : ' + name)
        entries[name] = digest
    if entries.get(SOURCE) not in (OLD_SHA, NEW_SHA):
        raise RuntimeError('Le manifeste ne correspond pas à la source attendue.')

    updates = {
        SOURCE: patched,
        'archive_repair/condensed_retraction_before_schur_rank_fix.cpp': original,
        'G1_SCHUR_NORMAL_RANK_FIX.md': NOTE.encode(),
    }
    archive_note = root / 'ARCHIVE_REPAIR_G1.md'
    banner = ('> Historique : ce document décrit la reconstruction avant le correctif\n'
              '> G1-SCHUR-RANK-1. Voir G1_SCHUR_NORMAL_RANK_FIX.md pour la modification\n'
              '> ultérieure de src/condensed_retraction.cpp.\n\n')
    if archive_note.is_file() and not archive_note.read_text().startswith(banner):
        updates[archive_note.name] = (banner + archive_note.read_text()).encode()
    install = root / 'INSTALL_AND_TEST_G1.md'
    if install.is_file():
        old = ("Le code du solveur et les quatre jeux de données sont\n"
               "identiques aux empreintes du manifeste G1 d'origine.")
        new = ("Les quatre jeux de données restent identiques aux empreintes G1.\n"
               "Le correctif du facteur normal est décrit dans G1_SCHUR_NORMAL_RANK_FIX.md.")
        text = install.read_text()
        if old in text:
            updates[install.name] = text.replace(old, new, 1).encode()

    # Verify every file to be changed before writing any of them. Other entries
    # keep their existing checksums and will be checked by sha256sum after patching.
    for name, data in updates.items():
        p = root / name
        if p.exists() and name != SOURCE:
            actual = sha(p.read_bytes())
            if actual not in (sha(data), entries.get(name)):
                raise RuntimeError('Fichier local modifié ; arrêt : ' + name)
    for name, data in updates.items():
        atomic_write(root / name, data)
        entries[name] = sha(data)
    manifest_bytes = ''.join(entries[name] + '  ' + name + '\n' for name in sorted(entries)).encode()
    atomic_write(manifest, manifest_bytes)
    print('PATCH_G1_SCHUR=' + ('ALREADY_APPLIED' if current_sha == NEW_SHA else 'APPLIED'), file=sys.stderr)
    print('SOURCE_SHA256=' + NEW_SHA, file=sys.stderr)
    print(root)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('PATCH_G1_SCHUR=FAIL ' + str(error), file=sys.stderr)
        raise SystemExit(1)
