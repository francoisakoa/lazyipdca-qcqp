EXÉCUTION DE L'ABLATION DU RAFFINEMENT

Les huit configurations de recherche sont FROZEN/REFINED × GR3/GHS ×
MA27/MUMPS. Elles utilisent la même initialisation hybride, l'objectif exact,
les mêmes garde-fous R3 et une représentation qui reste QCQP. FROZEN conserve
la métrique initialisée dans chaque phase ; REFINED autorise ensuite ses
mises à jour adaptatives. Ces étiquettes ne remplacent pas les profils R3,
HH, HS, SH, SS et SS_legacy de la campagne publiée.

Cette file lance 1 834 cas I/M × huit configurations = 14 672 résolutions
natives. Elle n'ajoute aucune tâche Ipopt. Toutes les cellules sont rejouées,
sans endpoint externe comme départ et sans substitution d'un ancien résultat.
Les limites natives sont 7 200 secondes, 5 000 tours, 600 restaurations
partagées entre toutes les phases. Un thread numérique est utilisé par tâche. L’ablation entière utilise le seul
cœur 23 et un seul créneau, avec un verrou distinct de celui d’Ipopt (0–21).

La commande de lancement générale du kit orchestre les étapes suivantes.
Les interfaces de bas niveau permettent également leur examen séparé.

1. Préparation, après la compilation isolée

python3 runtime/PREPARER_EXPERIENCE.py \
  --root INSTALLATION_SOURCE --run RUN_SOURCE \
  --build BUILD_EXPERIMENT.json --out DOSSIER_NEUF \
  --study-contract STUDY_CONTRACT.json

Les 4 078 entrées/départs du corpus source sont vérifiés, puis les 1 834 cas
I/M sont retenus sans changement. Les chemins déclarés et canonical_inputs
sont respectés ; aucun QCQP n'est copié ni régénéré. Le fichier de préparation
lie les sources, le plan, les bibliothèques, les commandes, le corpus et les
contrats par empreintes. Il porte production_authorized=false.

2. Qualification et scellement avant la production

Le programme qualification/QUALIFIER_EXPERIENCE.py doit produire un GATE50.json
complet sur l'échantillon fixé de 25 I et 25 M. Sur MA27, la référence
installée et sa reconstruction sont comparées sur les états finaux, compteurs
et audits. Sur les deux backends, les références reconstruites, leurs
observateurs passifs et les modes historiques de recherche doivent coïncider.
Les contrôles d'intervention doivent prouver les mêmes métriques initiales
et garde-fous, ainsi que la réalisation effective du mécanisme.
Un essai incomplet ou censuré ne valide pas la qualification.

python3 runtime/PREPARER_EXPERIENCE.py \
  --promouvoir DOSSIER_NEUF/PREPARATION.json \
  --qualification-gate CHEMIN/GATE50.json \
  --study-contract STUDY_CONTRACT.json

Le scellement vérifie aussi l'audit de séparation du code lié aux sources
compilées et le schéma d'instrumentation. La production lie ses observations
au nouveau PRODUCTION_REGISTRATION.json scellé, distinct du protocole statique
utilisé pour la qualification. EXPERIMENT_INSTALLATION.json n'autorise le
lancement qu'après l'ensemble de ces vérifications. Une répétition de la
promotion réutilise un reçu strictement identique et toujours valide.

3. File continue sur le cœur 23

python3 runtime/CONTROLEUR_EXPERIENCE.py lancer \
  --installation DOSSIER_NEUF/EXPERIMENT_INSTALLATION.json \
  --out SORTIE_NEUVE --interval 10

Le prochain travail occupe le créneau libéré sans attendre la fin d'un
dataset. L'audit KKT indépendant utilise le même créneau après le solveur ;
son temps est mesuré séparément. L'ordre des huit traitements est tournant
et équilibré. La graine ne modifie jamais les départs.

python3 runtime/CONTROLEUR_EXPERIENCE.py suivre --out SORTIE_NEUVE --interval 10
python3 runtime/CONTROLEUR_EXPERIENCE.py arret --out SORTIE_NEUVE
python3 runtime/CONTROLEUR_EXPERIENCE.py reprendre \
  --installation DOSSIER_NEUF/EXPERIMENT_INSTALLATION.json --out SORTIE_NEUVE

L'arrêt vise uniquement le contrôleur identifié et ses groupes de processus.
Aucun ancien résultat n'est effacé. Une reprise ne relance pas une résolution
qui dispose déjà d'un endpoint et de ses ressources ; elle peut terminer son
audit. Une preuve de ressources absente reste une observation technique
incomplète. Les résultats connus, traces, budgets et audits sont revérifiés.
Une violation persistée des garde-fous ou de la métrique initiale interdit la
reprise. Elle ne peut pas être oubliée par un redémarrage du contrôleur.
À globalisation et backend fixés, une seule des deux cellules qui atteint
la première métrique constitue aussi une violation d'intervention.

Qualification d'une observation

La qualification mathématique exige les trois résidus originaux au plus à
1e-6. La qualification expérimentale exige en plus les données/départs, les
budgets de processus ET natifs, le code, le backend et l'intervention admis.
Les traces finales sont vérifiées après acceptation : états, phases,
métriques initiales/courantes, objective E0=0, origines et compteurs réels.
La métrique FROZEN doit rester égale à son initialisation et n'exécuter aucune
tentative, mise à jour, acceptation ou restauration de raffinement.

Un retour natif avant la première métrique est admissible seulement avec le
marqueur de fin du solveur, zéro initialisation, zéro pas accepté, zéro travail
de raffinement et zéro factorisation Newton primaire, puis un audit original
valide. Le backend effectivement rapporté reste brut. Un tel retour ne prouve
pas qu'une factorisation MA27 ou MUMPS a été appelée.

Une interruption, un audit absent, un budget dépassé ou une trace insuffisante
reste distinct d'un résultat numérique non strict dans les preuves brutes.
Le suivi affiche certificats/terminées/prévues par régime et backend, et
signale les preuves incomplètes. Il détecte les états sauvegardés dont le
contrôleur n'est plus vivant, au lieu de les présenter comme un travail actif.
Le reçu final de l'intervention n'est positif que si toutes les traces et
tous les budgets natifs et de processus sont vérifiés, sans violation.

Tests locaux

python3 runtime/TEST_RUNTIME_EXPERIENCE.py

Ces tests synthétiques vérifient les refus de protocoles invalides, les
contraintes du facteur unique, les traces, les budgets, la reprise et la file
continue. Ils n'exécutent aucun solveur numérique et n'autorisent pas la
production à eux seuls. Le reçu réel TRX reste nécessaire.
