ABLATION DES BORNES IMPLIQUÉES — ASU — 10 NOVEMBRE 2026 — R2

OBJECTIF ET PÉRIMÈTRE
Comparer les mêmes 503 instances ASU avec R3, SS et HH, chacun avec et sans
retrait des bornes impliquées par les égalités quadratiques, sous MA27 et
MUMPS : 503 × 3 × 2 × 2 = 6 036 observations nouvelles. Les coefficients,
les x0 bruts, les budgets et l'audit du QCQP original sont conservés.
Les objectifs sont comparés uniquement sur des certificats communs.

Le patch agit uniquement sur le choix d'activer la règle de preuve existante
des bornes impliquées par les égalités quadratiques diagonales. L'option
--research-equality-implied-bounds profile|keep|remove conserve par défaut
le comportement de la source commune (HH actif, R3/SS inactifs). keep
désactive cette règle ; remove l'autorise également pour R3 et SS. Une règle
active ne force aucun retrait si sa preuve exacte échoue. Le traitement des
égalités affines singleton, celui des inégalités radiales, les garde-fous,
la globalisation et la levée des multiplicateurs ne sont pas modifiés.

Les rapports natifs exportent chaque côté réellement retiré et son origine.
Le contrôleur exige les comptes, les tableaux, leur cohérence et leurs
indices, ainsi que les dimensions originales du cas gelé. Ces 503 cas ne
comportent ni égalité affine ni coordonnée fixée : le compteur de retrait
par égalités identifie ici la règle quadratique étudiée. keep doit avoir
zéro retrait par égalités. Un retrait autorisé mais nul reste enregistré
comme tel ; aucun nombre théorique de bornes retirées n'est substitué.

SOURCE ET QUALIFICATION
Les 823 fichiers authentifiés de la source MA27 antérieure forment la base
commune. La compilation isolée produit deux binaires avec le même outil,
les mêmes options et dépendances : référence sans patch et version de
recherche. Chaque binaire contient les deux backends MA27 et MUMPS.
Le patch comporte quatre fichiers, dont un nouveau fichier d'aide.
La source historique MUMPS différait sur dix fichiers : aucune identité
numérique avec cette ancienne campagne MUMPS n'est revendiquée.

Avant l'ablation, 12 cas pilotes déterministes servent à 144 résolutions
de qualification (référence/recherche en mode profile, trois politiques,
deux backends). Les résultats, les compteurs de travail, les rapports de
présolve et l'audit numérique original doivent coïncider exactement.
Seuls les champs d'horloge dont le nom finit par _seconds sont exclus de
cette comparaison. Aucun seuil numérique relâché n'est utilisé. Chaque
commande, binaire, CPU, nombre de threads, reçu de ressources et artefact
est vérifié ; une différence ou une preuve manquante bloque la suite.
Cette qualification porte sur la base commune, sans prétendre reproduire
le MUMPS historique. Un certificat strict n'est pas exigé pour qualifier
la conservation du comportement ; les preuves et budgets doivent l'être.

ENCHAÎNEMENT ET REPRISE
STUDY_BOUNDS.py réalise construction, qualification, pilote d'ablation,
préparation des 503 cas, réutilisation des observations du pilote, campagne
complète et analyse. Le pilote d'ablation compte 144 observations. Sa
promotion exige des preuves complètes et au moins un retrait réel observé
par politique/backend. Il n'exige aucun bénéfice d'objectif ni taux KKT.
Ces 144 observations sont ensuite incluses dans les 6 036 sans les rejouer ;
il reste 5 892 nouvelles résolutions. Total prévu : 144 qualifications +
6 036 ablations = 6 180 exécutions natives, audits indépendants en plus.

Relancer exactement la même commande reprend les étapes manquantes. Les
observations finalisées et les censures restent enregistrées. Les endpoints
déjà produits sont audités sans nouveau solveur. La copie du pilote conserve
octet pour octet COMMAND, ressources, endpoint et audit ; un reçu explicite
lie l'observation initiale à sa place dans le plan complet. Une copie
interrompue est reconstruite en zone temporaire, sans résolution native.
Une modification du protocole, d'un binaire, d'une entrée ou d'un artefact
réutilisé provoque un refus. Une construction interrompue est conservée
pour inspection ; une nouvelle tentative isolée est créée automatiquement.

Ressources : cœurs 0–21, file continue de 22 créneaux, une thread numérique.
Budget par résolution : 7 200 s, 5 000 itérations extérieures, 600 itérations
de restauration partagées. Audit original indépendant : primal, dual et
complémentarité ≤ 10^-6 ; audit limité séparément à 600 s. Route AL native
désactivée ; backend réel vérifié, aucun fallback primaire pivoté admis.
La compilation est séquentielle sur le cœur 0. Les installations et
campagnes antérieures restent intactes. Le verrou CPU 0–21 est partagé avec
les kits antérieurs ; les travaux extérieurs au dispositif doivent aussi
avoir libéré ces ressources. Le kit n'arrête aucun programme existant.

FICHIERS DE SORTIE
PLAN_ETUDE.json : protocole immuable ; ETUDE_STATUS.json : étape ou blocage.
build/BUILD_BOUNDS.json : construction (ou build_retry_NNN après incident).
preparation_pilot/INSTALLATION.json ; qualification/QUALIFICATION.json.
pilote/ : 144 observations d'ablation et PILOT_GATE.json.
preparation_full/INSTALLATION.json ; campagne/ : plan des 6 036 observations,
dont les 144 réutilisées avec leurs reçus. analyse/PAIRED_SUMMARY.json et
analyse/PAIRED_OBJECTIVES.csv : dépouillement final. Voir COMMANDES.txt.

ANALYSE PRÉDÉFINIE
Chaque politique/backend a sa comparaison keep/remove : couverture des
certificats, objectifs appariés, médiane signée et médiane des écarts
absolus, quartiles, comptes aux seuils 10^-3 et 10^-2. Les cohortes communes
aux six cellules d'un backend sont aussi fournies. Les différences R3/SS/HH
à état de bornes fixé complètent l'analyse. Les nombres de côtés réellement
retirés sont exportés. Les objectifs ne sont comparés que lorsque le QCQP
original est certifié de part et d'autre. Les autres cas restent au
dénominateur de couverture et ne reçoivent aucun objectif inventé.

L'effet keep/remove isole la règle dans une politique et un backend ;
une comparaison entre politiques implique encore d'autres mécanismes.
L'étude permettra d'évaluer la contribution causale de ce retrait aux
écarts observés dans ce protocole. Aucun résultat causal ou gain de vitesse
n'est annoncé avant mesure. Ces temps proviennent d'une charge de 22 tâches ;
le chronométrage ciblé calme à trois répétitions est une étude distincte.
Aucun optimum global ni suivi exact d'une trajectoire centrale n'en découle.

VALIDATION LOCALE FOURNIE
73 vérifications C++ du choix générique profile/keep/remove ; six paires
de commandes identiques hors option ; tests de refus de preuves invalides,
comparaison exacte des champs non temporels ; tests synthétiques de reprise,
promotion, absence de répétition et rejet d'artefacts modifiés. Les tests
n'exécutent aucun solveur, même sur un cas jouet. La compilation complète,
les bibliothèques réelles et la qualification numérique restent à effectuer
sur TRX. Aucun résultat numérique natif local n'est revendiqué.

Les dépendances HSL ne sont pas redistribuées. BASE_CONTEXT.json conserve
les chemins historiques. Toute pièce absente ou différente est refusée ;
ne pas remplacer son empreinte par celle d'un autre artefact.
