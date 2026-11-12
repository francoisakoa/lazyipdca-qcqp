LAZYQCQP — ABLATION EXÉCUTABLE DE LA PERSISTANCE — 10 NOVEMBRE 2026 — R1

OBJECTIF
Mesurer le coût évité en conservant la courbure des majorants d'inégalités
de R3. Deux modes utilisent le même constructeur et les mêmes Hessiennes :
reuse conserve la correction ; rebuild la reconstruit avant chaque
nouvelle installation du modèle après la première. Une réinstallation au
même centre, notamment en restauration, compte aussi comme un événement.
La préparation initiale n'est pas effectuée deux fois au premier centre.

Le mode rebuild vide les données de courbure et leur correspondance dans
le modèle, puis appelle le constructeur original. Il conserve les états
primal-dual, les caches d'évaluation, les égalités, les paramètres et les
caches du solveur linéaire. Les compteurs qui influencent la continuation,
notamment le nombre courant de blocs spectraux positifs, sont préservés
mathématiquement. Le coût de reconstruction et de remise en place entre
intégralement dans le temps du processus solveur.

Ce contrôle porte sur R3 avec correction déterministe des inégalités,
convexification-study off et correction adaptative de l'objectif désactivée.
Il quantifie cette réutilisation de courbure. L'oubli d'atomes appris et la
stabilisation des égalités demanderaient d'autres interventions ; ils ne
sont pas modifiés. Le contrôle figé/adaptatif antérieur est distinct.

COHORTE ET NOMBRE DE RÉSOLUTIONS
12 instances fixées avant les nouvelles mesures, chacune avec MA27 et
MUMPS : 24 couples instance/backend. La sélection prend l'intersection
des cohortes historiques de chronométrage, avec des inégalités quadratiques,
six cas I et six cas M. Chaque régime est partagé en trois bandes de taille
et deux cas par bande sont choisis par empreinte avec une graine fixe.
BUILD_COHORTE_PERSISTANCE.py reproduit la liste COHORTE_PERSISTANCE.json.
La sélection n'utilise aucun gain observé dans cette nouvelle expérience.
Les succès historiques limitent la portée à cette cohorte ciblée.

Qualification : 24 × trois configurations référence/reuse/rebuild = 72 runs.
Chronométrage : 24 × deux modes × trois répétitions = 144 runs.
Total : 216 résolutions natives nominales, audits indépendants en plus.
Les tentatives interrompues ou perturbées sont conservées et ajoutent un
coût ; elles ne sont pas effacées des preuves. Les 72 contrôles tracés ne
sont pas réemployés comme répétitions de chronométrage.

CONSTRUCTION ET QUALIFICATION
Le code commun authentifié de la campagne MA27 est fourni dans
native/reference_source. Les sources historiques MUMPS différaient : cette
nouvelle expérience compare deux modes d'un même code, sous chaque backend.
Elle ne prétend pas reproduire numériquement cette ancienne version MUMPS.
BUILD_PERSISTANCE.py construit une référence intacte et une version de
recherche avec le même compilateur, les mêmes options et dépendances.
Les bibliothèques réelles HSL/MUMPS et les entrées restent celles de TRX.

Les contrôles comparent les endpoints et le travail numérique à la
référence, puis les deux modes entre eux. Ils comparent aussi les audits
originaux et l'ordre des empreintes SHA256 des corrections, états et
directions tracés. La représentation binaire64 distingue les zéros signés.
Les horloges et les deux compteurs de réutilisation directement modifiés
par l'intervention sont exclus explicitement des comparaisons appropriées.
Les paramètres numériques de continuation ne sont pas exclus.

Les événements tracés couvrent les recalages, les paramètres du contrôleur
et les pas acceptés de la boucle principale instrumentée ; ils ne couvrent
pas tous les essais rejetés et toutes les routines de secours internes.
Les points finaux et les compteurs numériques sont contrôlés en complément.
La trace et les empreintes de matrices sont désactivées pendant les temps.

Les rapports vérifient les constructions effectivement réalisées et les
réutilisations. Un cas sans modèle, sans correction non nulle ou sans
réinstallation reste dans le plan ; son absence d'information sur ce
mécanisme est signalée. Aucun nouveau cas n'est sélectionné après mesure.
Une preuve manquante ou une différence numérique bloque la qualification.

CHRONOMÉTRAGE ET INTERPRÉTATION
Un seul solveur à la fois, CPU 0, un thread numérique, processus neuf et
départ brut identique. Ordre équilibré, fixé avant les résolutions.
Budget : 7 200 s, 5 000 itérations extérieures, 600 itérations de restauration
partagées. La route AL est désactivée. Le backend effectif est contrôlé.
L'audit du QCQP original complet exige primal, dual et complémentarité
au plus 10^-6 ; il est séparé du temps solveur et limité à 600 s.

Le temps processus inclut chargement, préparation, reconstruction, calcul
et fermeture. Il exclut attente de calme et audit. Le moniteur observe tous
les CPU, y compris 22 et 23, et détecte aussi les charges CPU brèves.
Il attend dix secondes sous le seuil déclaré avant chaque résolution.
Il ne garantit pas l'absence de contention mémoire, disque ou thermique.
Une tentative polluée est conservée puis refaite après retour au calme.
Un échec numérique propre ou un budget épuisé reste une répétition acquise.
Aucune répétition n'est refaite pour obtenir un meilleur résultat.

Les comparaisons utilisent la médiane de trois répétitions propres et leur
dispersion. L'attribution au coût du cache exige aussi la concordance du
point final et du travail numérique des répétitions. Les divergences,
pertes de certification et preuves manquantes sont rapportées séparément.
Les rapports donnent les compteurs et temps de construction, les nombres
de corrections et les cas où le mécanisme est effectivement exercé.
Les ratios attribués au coût du cache exigent une construction effectivement
évitée lors de la qualification ; les cas inactifs restent décrits à part.
Un gain ne peut être annoncé qu'après lecture des nouvelles mesures.

EXÉCUTION, REPRISE ET COLLECTE
Voir COMMANDES.txt. LANCER_PERSISTANCE.sh enchaîne construction,
préparation, qualification, 144 temps et synthèse. Relancer la même commande
reprend les observations manquantes sans rejouer les résultats finalisés,
y compris les échecs numériques. Les preuves sont liées par empreintes.
Une ancienne résolution encore vivante après arrêt brutal du contrôleur
bloque un second lancement ; le PID et sa date de création sont contrôlés.

Les sorties sont dans un dossier distinct de ceux de l'ASU et des chronos
antérieurs. Ce kit ne modifie ni leurs installations ni leurs résultats.
Attendre la fin des autres calculs avant de lancer la campagne. Le kit
n'arrête aucun programme étranger et ne télécharge aucun nouveau dataset.
Les dépendances HSL ne sont pas redistribuées. Une dépendance ou une entrée
absente, ou une empreinte différente, provoque un refus avec son chemin.

Les validations locales livrées distinguent tests synthétiques, compilation
et contrôles numériques de petite taille. Elles ne remplacent pas les
72 contrôles sur TRX et ne constituent pas un résultat de vitesse sur cette
cohorte. Les expériences sur les 12 instances restent à exécuter sur TRX.
Les deux binaires locaux ont été entièrement compilés (40 unités et liaison).
Les 15 résolutions fonctionnelles finales MUMPS ont toutes obtenu un
certificat KKT original strict. La concordance des corrections, états et
compteurs a été vérifiée, y compris sur une correction spectrale et sur les
chemins sans construction. La validation numérique MA27 reste à faire sur TRX.
