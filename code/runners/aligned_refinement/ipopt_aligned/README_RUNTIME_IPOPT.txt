IPOPT : STRATÉGIES MONOTONE ET ADAPTATIVE, MA27 ET MUMPS

Cette campagne rejoue les 4 078 QCQP et leurs départs gelés dans quatre
configurations : monotone/MA27, monotone/MUMPS, adaptative/MA27 et
adaptative/MUMPS. Elle comporte donc 16 312 tâches et les quatre régimes
I, M, E et Q. Aucun résultat antérieur n'est réutilisé comme solution ou
point de départ. Les fichiers QCQP ne sont ni copiés ni régénérés.

Les tolérances déclarées sont tol=1e-8, constr_viol_tol=1e-6,
dual_inf_tol=1e-6 et compl_inf_tol=1e-6. L'arrêt acceptable est désactivé
(acceptable_iter=0) et bound_relax_factor=0. La stratégie de barrière et
le solveur linéaire constituent les quatre traitements. Les autres
réglages algorithmiques restent ceux de la bibliothèque liée. Le fichier
ipopt.opt ambiant est désactivé. Chaque endpoint conserve les options
effectives obtenues après initialisation de la bibliothèque.
Les seuls ajustements de représentation concernent les bornes finies qui
dépassent le seuil d'infini de la bibliothèque : elles restent finies dans
le QCQP transmis. Leurs seuils exacts sont déterminés par cas, vérifiés à
chaque endpoint et conservés dans les observations scalaires.

La préparation est indépendante des 748 essais de qualification du
raffinement. Elle exige la compilation isolée d'Ipopt et quatre contrôles
réels, un par combinaison stratégie/backend, avec audit analytique des
KKT originaux et un fichier ipopt.opt hostile ignoré. Une qualification
locale MUMPS seule n'autorise pas la production. Les dépendances TRX,
leurs bibliothèques transitives, les en-têtes et les sources sont liés par
empreintes et revérifiés. L'installation existante reste intacte.

Interfaces de bas niveau, également orchestrées par le lanceur général :

python3 ipopt_aligned/BUILD_IPOPT_ALIGNED.py --root SOURCE --out BUILD_NEUF
python3 ipopt_aligned/QUALIFIER_IPOPT_ALIGNE.py \
  --build BUILD_NEUF/BUILD.json --out QUALIFICATION_NEUVE
python3 ipopt_aligned/PREPARER_IPOPT_ALIGNE.py \
  --root SOURCE --run RUN_SOURCE --build BUILD_NEUF/BUILD.json \
  --qualification QUALIFICATION_NEUVE/QUALIFICATION.json --out PREPARATION_NEUVE
python3 ipopt_aligned/CONTROLEUR_IPOPT_ALIGNE.py lancer \
  --installation PREPARATION_NEUVE/INSTALLATION_IPOPT_ALIGNE.json \
  --out RESULTATS_NEUFS --interval 10

La file utilise 22 créneaux sur les cœurs 0 à 21, avec un thread numérique
par tâche. L'ablation native peut se dérouler en même temps sur le cœur 23.
Les verrous des deux ensembles de cœurs sont distincts. Une tâche suivante
occupe chaque créneau libéré sans attendre la fin d'un dataset. L'ordre des
traitements tourne de manière équilibrée ; aucun tirage ne modifie x0.
La priorité est CUTEst, Beamforming multicast, PGLIB, puis les autres.

Chaque tâche reçoit au plus 7 200 secondes CPU et murales et 5 000
itérations. Les budgets natifs et les mesures du processus doivent tous
être valides. Le temps et les ressources de l'audit sont enregistrés
séparément. Une réussite exige les trois résidus KKT originaux au plus à
1e-6, recalculés indépendamment sur le QCQP complet. Le statut propre
d'Ipopt et le certificat original restent des informations distinctes.

python3 ipopt_aligned/CONTROLEUR_IPOPT_ALIGNE.py suivre \
  --out RESULTATS_NEUFS --interval 10

Le suivi compact montre quatre lignes de régimes et quatre colonnes de
configurations, les totaux, les créneaux solveurs/audits et les datasets
actifs. --detail ajoute le détail des datasets ; --once imprime une photo.

python3 ipopt_aligned/CONTROLEUR_IPOPT_ALIGNE.py arret --out RESULTATS_NEUFS
python3 ipopt_aligned/CONTROLEUR_IPOPT_ALIGNE.py reprendre \
  --installation PREPARATION_NEUVE/INSTALLATION_IPOPT_ALIGNE.json \
  --out RESULTATS_NEUFS --interval 10

L'arrêt ne vise que le contrôleur identifié et ses enfants. Aucun ancien
résultat n'est effacé. Une reprise peut terminer l'audit d'un endpoint
disposant de ses ressources ; elle ne relance pas son optimiseur. Les
preuves des résultats finalisés sont revérifiées. Un état sauvegardé avec
un contrôleur mort n'est pas présenté comme un calcul actif. Une preuve
manquante reste séparée d'un échec numérique dans les observations brutes.

PLAN.json lie le plan contemporain. Chaque tâche conserve sa commande,
son journal, son endpoint avec les vecteurs primal/duaux et options, les
ressources solveur, l'audit indépendant, ses ressources et RESULT.json.
RESULTS.jsonl et OBSERVATIONS.jsonl rendent les observations accessibles
au dépouillement. Le second fichier est reconstruit depuis les reçus de
tâches à la reprise pour récupérer une écriture de journal interrompue.

Vérifications locales :

python3 ipopt_aligned/TEST_RUNTIME_IPOPT_ALIGNE.py

Ces tests vérifient les refus de paramètres incohérents, les budgets, les
chemins gelés, la file continue, les états morts et la reprise. Leur test
d'intégration porte sur la liste réelle des 4 078 identifiants avec un
résolveur simulé. Ils ne lancent aucun solveur et ne remplacent pas les
quatre contrôles numériques exécutés sur TRX.
