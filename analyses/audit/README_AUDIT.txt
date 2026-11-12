AUDIT DE L'ARCHIVE IPOPT ALIGNÉE — 10 novembre 2026

ENTRÉES ET PORTÉE
Archive : IPOPT_ALIGNE_A_TRANSMETTRE_20261110T015338_235cbb1c.zip.
16 312 tâches distinctes = 4 078 mêmes QCQP et départs bruts × quatre
configurations (MA27/MUMPS ; monotone/adaptive). Les observations sont
appariées à la matrice planifiée et aux douze configurations natives
du compact de la campagne précédente. Aucun résultat Ipopt à arrêt par
défaut n'est utilisé comme référence principale de cette analyse.

Pour chacune des 16 312 tâches, les métadonnées RESULT, case, commande et
ressources sont confrontées. Pour les 16 213 tâches disposant d'un AUDIT,
le contenu indépendant archivé est confronté à sa copie dans RESULT,
aux empreintes d'entrée et d'endpoint, et aux trois résidus originaux.
Les options effectives, leurs empreintes, la version 3.14.11, le backend,
la stratégie, le départ et les limites sont vérifiés pour les observations
complètes. Les 12 991 certificats stricts et les 3 130 observations non
strictes ont tous ces éléments cohérents. Aucun certificat n'est modifié.
Les 191 autres résultats restent séparés des observations auditables.

Limite : l'archive ne livre aucun endpoint brut de campagne, aucun journal
solver.log de campagne, ni les coefficients QCQP gelés. L'audit présent
contrôle donc les rapports indépendants et preuves archivés ; il n'évalue
pas de nouveau les résidus à partir des coefficients et multiplicateurs.
Les quatre endpoints et journaux du petit test de qualification initial
sont présents, mais ne représentent pas les 16 312 résolutions du corpus.

RÉSULTATS STRICTS RECOMPTÉS — I / M / E / Q / TOTAL
MA27 monotone : 1008 / 438 / 1485 / 534 / 3465.
MUMPS monotone : 1007 / 423 / 1153 / 536 / 3119.
MA27 adaptive : 989 / 336 / 1373 / 537 / 3235.
MUMPS adaptive : 989 / 310 / 1336 / 537 / 3172.
Dénominateurs : 1037 / 797 / 1704 / 540 / 4078.

Les 191 observations techniquement non qualifiées comprennent :
- 92 retours Not_Enough_Degrees_Of_Freedom (23 dans chaque configuration) ;
- 87 interruptions externes à la limite de temps de 7 200 secondes ;
- 12 terminaisons du processus par SIGSEGV, sans endpoint disponible.
La matrice de tâches est complète ; toutes ces tâches restent dans les
dénominateurs de certification. Leurs causes restent distinctes.

STATUTS INTERNES
Solve_Succeeded = code 0 ; Solved_To_Acceptable_Level = code 1.
Les codes non négatifs 2 et 3 (infeasibility/tiny search direction) ne sont
pas des succès. COVERAGE_RECOMPUTED.csv distingue : succès 0/1, tout retour
non négatif, certificat original et croisements entre ces catégories.
Le champ ipopt_own_success du contrôleur initial compte le code 0 seul.
Il diffère donc de la réunion 0/1 sur 577 retours acceptables. Il s'agit
d'une différence de définition du suivi, sans effet sur les certificats.
La même vérification sur l'ancienne campagne à arrêt par défaut confirme
les 3 787/3 749 retours 0/1 et les 1 859/741 retours 0/1 non certifiés.
Ces chiffres anciens ne comptent donc pas les codes 2 ou 3 comme succès.

Acceptable_iter=0 a été vérifié dans les options effectives ; 577 retours
code 1 sont néanmoins enregistrés. Le code officiel épinglé 3.14.11 explique
cette possibilité : IpOptErrorConvCheck.cpp:249 contrôle l'heuristique des
itérés acceptables successifs seulement si acceptable_iter_>0 ;
IpBacktrackingLineSearch.cpp:558–560, 579–580 et 606–607 peut lever
ACCEPTABLE_POINT_REACHED à l'entrée ou à l'échec de restauration, y compris
après restauration d'un point acceptable sauvegardé. Ce chemin ne teste pas
acceptable_iter_. IpIpoptAlg.cpp:469–478 traduit cette exception en arrêt
acceptable. Il ne faut donc pas présenter acceptable_iter=0 comme supprimant
tout retour acceptable. Les journaux absents ne permettent pas d'identifier
la branche exacte de chacun des 577 retours observés.

Un autre chemin de la même version assouplit les tests de dualité et
complémentarité pour un problème interne carré. Le drapeau correspondant
existe dans l'endpoint de l'adaptateur, mais n'est pas conservé dans les
RESULT/AUDIT de campagne de cette archive. Aucune cause de non-certification
sur PGLib, GasLib ou ASU ne lui est donc attribuée ici.

Les valeurs numériques de tolérance sont alignées ; les définitions
internes et le résidu original indépendant restent distincts. Les tableaux
de causes décrivent les composantes originales dépassant 10^-6, sans en
déduire une faute d'algorithme ni une causalité non observée.

REPRODUCTION
python3 audit_aligned_archive.py --archive /chemin/archive.zip \
  --native /chemin/COMPACT_14_CONFIGS.jsonl.gz --out /chemin/audit
Ce script utilise seulement la bibliothèque standard Python et ne lance
aucun optimiseur, aucune évaluation d'endpoint et aucune requête réseau.
Les sources Ipopt officielles épinglées utilisées pour l'interprétation
des statuts sont identifiées séparément avec leurs empreintes.
