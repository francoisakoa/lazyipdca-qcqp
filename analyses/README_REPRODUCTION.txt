LAZYQCQP JOGO R19 — ANALYSES PORTABLES

Python 3.12 ; numpy pour les références ; matplotlib pour les figures.
Depuis ce dossier :
  python3 run_reproduction.py --out reproduced

Cette commande reproduit les tableaux appariés des seize configurations,
les références publiées, les contrôles natifs et les tableaux complets du
manuscrit. Elle vérifie les sorties TeX et les 630 résumés appariés contre
les sorties archivées. Aucun optimiseur ni accès réseau n'est lancé.

Les 65 248 observations sont contenues une seule fois dans
pairs/OBSERVATIONS_16_CONFIGS.jsonl.gz. Les 8 156 observations distinctes
des deux références à arrêt par défaut servent au diagnostic historique
d'arrêt, sans modifier la comparaison principale monotone alignée.
Toutes les comparaisons natives principales sont conservées, avec MA27
et MUMPS. Les dossiers techniques gardent leur cause propre dans les
données ; les taux du manuscrit portent sur l'obtention du certificat.

Audit optionnel des preuves de l'archive fournie séparément :
  python3 run_reproduction.py --out reproduced --archive /chemin/IPOPT_ALIGNE_A_TRANSMETTRE_20261110T015338_235cbb1c.zip
Cet audit contrôle les rapports indépendants archivés, options, budgets,
empreintes et liaisons. Il ne recalcule pas les résidus à partir du QCQP :
les coefficients et endpoints bruts de campagne ne sont pas dans l'archive.

Profils empiriques, à partir des mêmes données classifiées :
  python3 manuscript_tools/plot_aligned_profiles.py --compact pairs/OBSERVATIONS_16_CONFIGS.jsonl.gz --out reproduced/figures
  python3 manuscript_tools/plot_aligned_profiles.py --compact pairs/OBSERVATIONS_16_CONFIGS.jsonl.gz --out reproduced/figures --regimes E Q --stem aligned_equality_performance_profiles
Les sept courbes d'un panneau sont les six politiques natives et la
référence monotone du même backend. Tous les cas prévus entrent dans
le dénominateur ; un cas sans certificat a un temps infini.

Les références QPLIB et MINLPLib sont des points publiés admis après
contrôle du sens et de l'échelle de l'objectif. Elles ne sont pas présentées
comme des optima globaux démontrés. Aucune référence M n'est créée.

L’expérience croisée terminée (14 672 observations) est documentée dans
refinement/. Les 189 observations techniquement non qualifiées restent
séparées, et le contrôle final des interventions conserve passed=false.
La qualification préalable de 748 résolutions était validée. Les tables
et analyses conservent les classifications fournies ; aucun nouveau
calcul KKT ni solveur n’est lancé. Voir refinement/README.txt.

Reproduction du nouveau volet depuis ce dossier :
  python3 refinement/analyze_refinement_pairs.py --compact refinement/REFINEMENT_OBSERVATIONS_8_CONFIGS.jsonl.gz --study refinement/STUDY.json --out reproduced/refinement
  python3 refinement/render_compact_tables.py --summary reproduced/refinement/REFINEMENT_PAIR_SUMMARY.json --out reproduced/refinement

Les contrôles complets de l’archive et les diagnostics des observations
indéterminées requièrent l’archive utilisateur conservée séparément :
RAFFINEMENT_TERMINE_20261110T121503_184817Z.zip.
Les rapports et leurs scripts sont dans refinement/archive_integrity/
et refinement/indeterminate/. Le tableau original de production, le
contrat scellé et le verdict expérimental sont fournis dans refinement/.


R18 — ASU et persistance
-----------------------
La commande run_reproduction.py inclut maintenant campaigns_R18/recompute_campaigns.py.
Ce dernier peut aussi être exécuté seul avec --out derived. Il reproduit les
6036 observations/3018 paires ASU et les 24 paires de persistance à partir des
scalaires, des empreintes et des contrôles enregistrés. Voir campaigns_R18/README.txt.
Aucun solveur ni nouvel audit de vecteur n’est lancé. Les chiffres du benchmark
principal et des comparaisons Ipopt à arrêt aligné ne sont pas modifiés.
