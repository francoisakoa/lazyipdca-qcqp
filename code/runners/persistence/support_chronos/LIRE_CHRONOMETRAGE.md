Ce dossier prépare le chronométrage ciblé R3 contre Ipopt aligné à stratégie monotone. Il réutilise les binaires historiques, sans recompilation, et les QCQP et départs gelés d’origine. Il ne lance aucun solveur pendant la préparation ou la synthèse.

Les quatre cohortes sont fixées avant les nouvelles mesures : certificat original strict pour les deux configurations et **deux temps historiques strictement supérieurs à une seconde**. Les instances et configurations ne seront pas sélectionnées d’après les nouvelles durées.

| Cohorte | Paires |
|---|---:|
| MA27, I | 30 |
| MA27, M | 89 |
| MUMPS, I | 81 |
| MUMPS, M | 75 |
| Total | 275 |

Cela représente 179 instances distinctes, 550 couples instance/configuration distincts, puis **1 650 exécutions pour trois répétitions**. Les durées historiques de ces 550 couples totalisent environ 7,95 h, soit 23,84 h en les multipliant par trois. Ce calcul n’est pas une prédiction de la durée hors charge : les attentes de calme, audits et éventuelles tentatives contaminées s’ajoutent.

Exécuter sur la machine TRX qui conserve les installations et résultats d’origine. Python 3.11 ou ultérieur, Linux `/proc`, le cœur 0 et les dépendances historiques sont nécessaires. Aucune bibliothèque Python tierce n’est requise.

```bash
python3 CHRONO_CALME.py preparer --out /extra/LAZYQCQP_CHRONO_CALME_RESULTS/CHRONO_20261110
python3 CHRONO_CALME.py lancer --out /extra/LAZYQCQP_CHRONO_CALME_RESULTS/CHRONO_20261110
python3 CHRONO_CALME.py synthese --out /extra/LAZYQCQP_CHRONO_CALME_RESULTS/CHRONO_20261110
```

La première commande exige un dossier absent ou strictement vide. Elle reprend ainsi une préparation interrompue juste après la création du dossier. Elle refuse tout dossier non vide, y compris un fichier temporaire d’installation incomplet, et en conserve le contenu. Elle vérifie les empreintes des binaires, bibliothèques, reçus de qualification, entrées et lignes `x0`, puis écrit le plan immuable `INSTALLATION.json`. Le résultat Ipopt aligné recherché par défaut est `/extra/LAZYQCQP_IPOPT_ALIGNE_RESULTS/IPOPT_4CONFIG_20261109T140336_fa503d1c`. L’option `--ipopt-run` de `preparer` permet de préciser son dossier existant ; elle n’autorise pas un autre binaire. Un fichier absent ou modifié arrête la préparation avec son chemin. Les fichiers historiques restent utilisés à leurs emplacements d’origine.

Le contrôleur lance une seule résolution à la fois, sur le cœur 0, avec un seul fil numérique, 7 200 s et 5 000 itérations au maximum. Chaque répétition démarre dans un processus neuf depuis le départ brut gelé. L’ordre est déterministe et inverse les deux configurations entre les deux premières répétitions. La troisième conserve un ordre fixé avec rotation des instances. Il ne réutilise aucun point final comme départ.

La commande native conserve le profil historique R3 et ses politiques de campagne : `--native-al-fallback off`, `--standard-r10-policy auto`, bornes impliquées activées, budget de restauration commun 600. Pour MA27, le chemin HSL et le rapport de présolve correspondent à l’exécuteur historique. Pour MUMPS natif, le CLI reçoit `mumps` et `--linear-metis 1` ; l’étiquette effective est `mumps_metis`. Ipopt conserve son ordre MUMPS automatique effectif 7 et toutes les options alignées qualifiées. Aucun nouveau réglage de pivot, mise à l’échelle ou borne n’est ajouté.

Avant chaque résolution, le contrôleur attend dix secondes consécutives sous le seuil de charge déclaré. Pendant la résolution, il observe chaque seconde les processus et les compteurs CPU agrégés de **tous les cœurs, y compris 22 et 23**. Le seuil est 0,05 seconde CPU étrangère par seconde écoulée, avec une marge de 0,03 s pour la granularité des compteurs. Les fenêtres individuelles détectent également une activité brève qui serait diluée dans la moyenne d’une longue résolution ; le temps CPU final du solveur provient de `wait4`. Les seuils et observations sont enregistrés. Cette observation mesure la concurrence CPU ; elle ne garantit pas l’absence de contention mémoire ou d’entrées-sorties.

Une tentative contaminée est conservée dans `runs/<répétition>/attempt_####/`, avec ses journaux, temps, endpoint et audit. Elle ne remplit pas la répétition prévue. Le contrôleur attend de nouveau le calme avant de refaire cette même répétition. Il n’arrête aucun programme étranger. Une tentative propre qui échoue numériquement ou épuise son budget **remplit la répétition** et n’est pas relancée pour obtenir un meilleur résultat. Toutes les tentatives sont comptabilisées dans la synthèse.

Le chronomètre mesure l’enveloppe commune du processus direct neuf, du lancement jusqu’à sa récupération par `wait4`, avec une interrogation de fin toutes les 5 ms. Il comprend chargement, initialisation et fermeture. L’audit indépendant du QCQP original, à tolérance `1e-6`, est lancé ensuite et reste hors du temps solveur. Le contrôleur attend sa fin avant de préparer la résolution suivante. Les enveloppes historiques utilisaient des démarrages différents selon leur producteur ; aucune durée de démarrage historique n’est soustraite.

Pour suivre, consulter `STATUS.json`, les messages de la commande et le fichier `QUIET.jsonl` indiqué pour la tentative active. Une attente initiale peut durer tant qu’une autre campagne travaille. Pour reprendre après interruption, réexécuter **la même commande `lancer --out ...`**. Les répétitions déjà validées, y compris les échecs numériques, ne sont pas refaites. Une tentative propre finalisée avant une coupure est récupérée même si l’écriture de `RESULT.json` n’avait pas eu lieu. Les tentatives incomplètes sont conservées séparément. Un verrou interdit deux contrôleurs de chronométrage simultanés pour le même utilisateur. Après une coupure brutale du contrôleur, une ancienne résolution encore vivante bloque la reprise jusqu’à sa fin ; son PID et son heure de création sont vérifiés pour éviter les confusions de PID.

La synthèse produit `SYNTHESE_CHRONO.json`, `PAIRES_CHRONO.csv` et `OBSERVATIONS_CHRONO.csv`. Elle utilise la médiane des **trois** répétitions propres de chaque couple instance/configuration, rapporte les ratios appariés R3/Ipopt et leurs quartiles, puis le nombre de paires toujours certifiées sur les trois répétitions. Elle conserve les pertes de certification, dépassements de budget et preuves indisponibles. Les cas incomplets ne reçoivent pas de médiane. Les minima et maxima sont descriptifs ; le meilleur temps n’est jamais la valeur comparée. Le bilan de temps comprend également les tentatives contaminées et audits disponibles. Un arrêt avant l’écriture des ressources reste signalé comme coût incomplet, sans estimation inventée.

Les conclusions de vitesse portent sur ces cohortes I/M sélectionnées par les données historiques ; elles ne constituent pas une comparaison de vitesse sur les 4 078 instances.

Vérification synthétique, sans solveur QCQP :

```bash
python3 TEST_CHRONO_CALME.py
python3 cohort/TEST_COHORT.py
```
