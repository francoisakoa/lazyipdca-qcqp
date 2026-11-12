MÉGA CAMPAGNE — DISPOSITIF PRÉCÉDENT, LAZY MA27 ET IPOPT PAR DÉFAUT
8 novembre 2026

DÉMARRAGE PRINCIPAL

Extraire l'archive dans Téléchargements puis :
  bash LANCER_MEGA.sh
  bash SUIVRE.sh

La première commande prépare puis lance automatiquement le contrôleur en
arrière-plan. La seconde affiche un suivi compact I/M/E/Q toutes les 10 s.
Ctrl+C ferme seulement le suivi. Pour arrêter les calculs : bash ARRETER.sh.

PÉRIMÈTRE

4 078 instances, 28 datasets, sans Dey, tous les régimes présents :
I=1 037, M=797, E=1 704, Q=540. Il n'y a pas de population A.

Six profils Lazy : R3, HH, HS, SH, SS, SS_legacy, tous avec MA27.
Deux références Ipopt avec les paramètres algorithmiques de bibliothèque :
MA27 et MUMPS. Total principal : 4 078 × 8 = 32 624 tâches.

22 créneaux sur les cœurs 0 à 21. Priorité CUTEst → multicast Beamforming →
PGLIB → autres. Les cœurs libres prennent immédiatement la prochaine tâche,
sans attendre la fin d'un dataset. Le contrôleur, ses audits et ses reçus
proviennent du dispositif de la campagne précédente : 32 fichiers récupérés
correspondent à leurs empreintes publiées. Les adaptations sont documentées.

DONNÉES ET CORRECTION DU BLOCAGE

Source existante, utilisée en lecture seule :
/data/projects/IPDCA_QCQP/LAZYQCQP_MEGA_STABILISEE_20261107_R2

Le premier kit utilisait à tort data/frozen/<identifiant>/input.qcq pour tous
les cas. Or la campagne précédente avait conservé la sérialisation brute puis
référencé les 120 entrées Beamforming depuis canonical_inputs/. Le nouveau
lancement reprend PREPARATION.json et ses chemins déclarés, et vérifie les
empreintes publiées ainsi que le x0 brut. Il ne recanonicalise ni ne régénère
les QCQP. Aucun endpoint externe ne sert de départ.

Les gros fichiers gelés sont utilisés en place : aucune nouvelle copie des
~22 Go de données. Les nouvelles compilations, preuves et installations vont
sous /extra/LAZYQCQP_MEGA_MA27_INSTALL ; les nouveaux résultats sous
/extra/LAZYQCQP_MEGA_MA27_RESULTS. L'installation et les résultats précédents
ne sont ni effacés ni remplacés. La source existante reste nécessaire.

MA27 : MODIFICATION DÉCLARÉE ET QUALIFICATION

L'ancien code réservait certains chemins à MUMPS. Il ne suffisait donc pas de
changer un argument du lanceur. Une adaptation isolée fournit les capacités
MA27 requises : admission des politiques, inertie déjà calculée remontée aux
contrôles existants, classification de courbure, restauration avec limites de
mémoire A/IW et étiquette de backend correcte. Les conditions d'acceptation,
seuils de pivot et raffinement restent ceux du code de référence.

Les sources qualifiées d'origine sont conservées. La nouvelle référence MA27,
son observateur passif et le candidat croisé sont compilés séparément avec
les mêmes options et dépendances. Les exécutables natifs exposent MA27 et
omettent MUMPS ; aucune résolution native n'est transférée à Ipopt. La vraie
bibliothèque HSL de l'installation est utilisée et liée par SHA256, jamais
incluse dans l'archive.

Avant de lancer la méga : 24 essais réels MA27 (six profils × I/M/E/Q), avec
facteurs MA27 réussis et audit original strict ; qualification séparée des
quatre contrôles Ipopt pour chacun de ses deux backends ; contrôle complet
des lecteurs, bornes, certificats et entrées du corpus. Un contrôle invalide
bloque le lancement avec sa cause et son journal.

La compilation locale de 20 unités C++ réussit. Les tests de contrôleur,
d'adaptation et d'audit réussissent. Aucune réussite numérique MA27 sur TRX
n'est annoncée dans ce kit : HSL réel et les qualifications sont exécutés
sur la machine utilisateur. Le résultat CUTEst historique 132/136 sous un
autre backend ne constitue pas une qualification du nouveau run MA27.

IPOPT PAR DÉFAUT

La campagne enregistrée utilisait explicitement adaptive. Le nouveau binaire
conserve les options algorithmiques de la bibliothèque 3.14.11, dont monotone,
acceptable_iter=15 et l'ordre MUMPS automatique 7. Les valeurs effectives sont
lues après initialisation et contrôlées avant lancement puis à chaque tâche.
Sont déclarés séparément le backend, la bibliothèque HSL, les budgets, les
sorties et l'encodage d'une borne finie dépassant le seuil d'infini de l'API.
Un ipopt.opt externe n'est pas chargé.

Qualification commune : primal, dual et complémentarité sur le QCQP original
complet, tous <= 1e-6. Un statut interne solved/acceptable ne suffit pas.
Les observations manquantes, interrompues ou non auditables restent inconnues.
Budgets par résolution : 7 200 s, 5 000 itérations extérieures, 600 restaurations
natives partagées entre les phases. Les audits indépendants sont chronométrés
séparément. Temps internes, factorisations, résolutions et ressources wait4
sont conservés. Les horloges qui se recouvrent ne sont pas additionnées.

EXPÉRIENCE CROISÉE CONSERVÉE

Lancement alternatif : bash LANCER_TOUT.sh.

Ce mode ajoute 7 336 tâches natives aux 32 624 principales, soit 39 960 dans
la MÊME file de 22 créneaux. Les quatre cellules I/M, avec objectif exact,
sont F_R3, F_X1, F_X2, F_H0. Elles restent distinctes des profils historiques,
y compris le R3 historique. F_H0 n'est pas rebaptisé HH.

La préparation supplémentaire exige le contrôle de séparation des facteurs
sur les 1 834 modèles et la reproduction comparative sur les 50 cas fixés,
avec une référence nouvellement compilée MA27 des deux côtés. Une ancienne
qualification MUMPS ne vaut pas pour MA27. Un couplage signé, une censure ou
une preuve manquante interdit le mode croisé. Le mode principal LANCER_MEGA
n'est pas conditionné à ces contrôles propres à l'expérience.

Le préenregistrement MA27 conserve les entrées, le x0, l'échantillon de 50,
les seuils et les critères annoncés (gain >=20, discordances >=2:1,
concentration <=50% du gain net, signe positif en pondération par famille).
L'amendement de backend est déclaré avant les nouveaux résultats. L'ordre
des traitements est équilibré par rotations déterministes ; la graine
n'affecte jamais un point de départ du solveur.

La provenance FINALE des directions acceptées est tracée pour les quatre
cellules expérimentales. Elle ne doit pas être imputée aux six profils
historiques, dont les compteurs restent ceux du dispositif précédent.
Choisir un seul mode de lancement. Le verrou global empêche deux contrôleurs
respectant ce dispositif de réserver simultanément les 22 créneaux.

DÉPOUILLEMENT

  bash DEPOUILLER.sh

Produit un ZIP avec toutes les variantes, résidus originaux, statuts propres,
objectifs aux seuils symétriques 1e-4/1e-3/1e-2, temps appariés (dont les deux
processus >1 s), pondérations par instance/famille/dataset et preuves compactes.
Les quatre cellules croisées sont analysées séparément, selon le protocole
préenregistré. Aucun ancien résultat n'est ajouté au nouveau run.

  bash REPRENDRE.sh

Reprend le même plan après vérification des identités ; les observations
finalisées ne sont pas résolues une seconde fois. Les contrôleurs absents
sont signalés immédiatement, sans afficher des créneaux périmés pendant 24 h.

PRÉREQUIS

Linux, Python >=3.11, g++, Boost et les en-têtes/dépendances de la précédente
installation, bibliothèque HSL existante. Si la source est ailleurs :
  bash LANCER_MEGA.sh /chemin/vers/installation_precedente
Aucune mise à niveau du système, récupération réseau de datasets ou licence
HSL de remplacement n'est exécutée par les scripts.
