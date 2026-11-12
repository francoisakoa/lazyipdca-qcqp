QUALIFICATION AVANT L'EXPÉRIENCE DE RAFFINEMENT

Le script QUALIFIER_EXPERIENCE.py consomme la préparation en lecture seule,
le manifeste des trois binaires fraîchement compilés et le contrat scientifique
statique. Il ne choisit aucun cas selon les résultats des solveurs.

Commande depuis le kit :
python3 qualification/QUALIFIER_EXPERIENCE.py \
  --preparation /chemin/isole/PREPARATION.json \
  --build /chemin/isole/build/BUILD_EXPERIMENT.json \
  --study-contract STUDY_CONTRACT.json \
  --out /chemin/isole/qualification --cpus 23 --jobs 1

La sortie doit être neuve. Le lanceur principal orchestre ces étapes.
Aucun résultat, binaire ou QCQP existant n'est supprimé ni modifié.

Le contrôle comprend 748 exécutions natives :
- 50 cas originaux fixés par le contrat, 25 du régime I et 25 du régime M ;
- R3 et HH dans leurs profils complets, sous MA27 et MUMPS ;
- référence sans observateur, référence avec observateur passif, puis binaire
  de recherche sans activer les nouveaux facteurs : 600 exécutions ;
- sous MA27, le binaire installé de la campagne précédente est aussi exécuté
  sur les 50 cas et les deux profils : 100 exécutions. Ses vecteurs,
  compteurs, statuts et audits sont comparés à la référence reconstruite.
  Aucune trace absente du binaire installé n'est inventée ;
- six contrôles structurels génériques, les quatre cellules exactes et deux
  backends : 48 exécutions supplémentaires.

Les limites natives restent 7 200 s, 5 000 itérations extérieures et 600
restaurations partagées. Le temps du processus externe et le temps natif
doivent chacun respecter la borne. La file commune alimente un créneau sur CPU23, distinct des cœurs 0–21
réservés à Ipopt.
Le suivi affiche l'avancement au moins toutes les dix secondes.

La fidélité exige les mêmes états binaires64 engagés, les mêmes vecteurs de
sortie, compteurs non temporels et décisions d'audit KKT original. Seuls les
champs de temps sont exclus de la comparaison d'identité. Un arrêt complet à
la limite d'itérations peut être comparé exactement ; un arrêt au temps, un
signal, une trace ou un endpoint manquant reste non qualifiable et bloque
l'autorisation. Les échecs numériques identiques des deux implémentations
n'invalident pas leur identité. Aucun rééchantillonnage n'est permis après
lecture d'un résultat.

Les contrôles de l'intervention vérifient l'objectif exact, la même première
correction dans chaque paire gelé/raffiné, le même vecteur de garde-fous,
la représentation QCQP originale et l'absence complète de travail de
raffinement dans les cellules gelées. Les matrices courantes sont vérifiées
aux engagements effectifs, avec leurs phases. Une phase résolue avant de
construire un modèle est enregistrée comme telle ; aucun modèle ni travail
n'est inventé. Chaque contrôle structurel doit atteindre un modèle et engager un pas
avec une véritable factorisation du backend déclaré. Pour chaque
globalisation et backend, au moins un contrôle raffiné doit effectuer une
tentative adaptative réelle. Un gain positif ou un raffinement accepté n'est pas une condition
de qualification : ce sont précisément des résultats de l'expérience.

Le reçu GATE50.json ne porte passed=true et production_authorized=true
qu'après tous les contrôles réels. Il lie les données et départs résolus,
les binaires, le contrat et la préparation par empreinte. Les traces détaillées
restent à côté des comparaisons et audits. Le contrat de production est scellé
ensuite, en liant ce reçu ; aucun cycle d'empreintes n'est créé.

Validation locale du contrôleur :
python3 qualification/TEST_QUALIFIER_EXPERIENCE.py
Ces tests synthétiques ne lancent aucun optimiseur et ne constituent jamais
la qualification réelle des 50 cas sur TRX.
