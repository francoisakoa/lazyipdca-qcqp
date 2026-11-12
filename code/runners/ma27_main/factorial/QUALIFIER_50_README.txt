QUALIFICATION NATIVE FACTORIELLE — CONTRÔLE AVANT CAMPAGNE

Commande (les chemins de construction doivent désigner les reçus réels TRX) :
python3 QUALIFIER_50.py --root ROOT --binary RESEARCH_BINARY \
  --baseline-trace-binary OBSERVER_BINARY --build-proof BUILD_NATIVE.json \
  --coupling-proof COUPLAGE_COMPLET.json \
  --registration PRE_ENREGISTREMENT.json --out FRESH_DIRECTORY \
  --jobs 22 --cpus 0-21

Avant ces essais, un reçu indépendant doit vérifier les 1 834 entrées I/M
(1 037 I et 797 M) : aucune réduction par carrés signés ne doit changer la
représentation. Toute entrée manquante, transformée ou non résolue bloque
le lancement ; le reçu est lié au binaire, aux sources et au préenregistrement.

Les 50 entrées sont fixées avant lecture des résultats : 25 I et 25 M.
Pour chaque entrée : R3 historique et HH historique sont chacun exécutés avec
le binaire publié inchangé, puis son observateur passif, puis la diagonale de
recherche (objectif exact pour R3, objectif historique pour HH). Cela donne
300 exécutions réelles et 100 comparaisons des séquences complètes d'états.
Les trois autres cellules à objectif exact sont également exécutées sur deux
entrées structurellement choisies du même échantillon, soit 6 exécutions.

Les profils de recherche proviennent tous du même modèle de commande HH,
avec les deux facteurs explicites et accepted-checkpoint=0. La fonction
PROFILES.make_command est commune à la qualification et à la production.
L'ancien HH avec objectif corrigé est réservé au contrôle diagonal ; H0
reste la cellule hybride/hs à objectif exact dans la campagne.

Les seules différences séparées du contrôle numérique sur la diagonale HH
sont le compteur d'observation free_epigraph_checkpoint_lifts et les quatre
métadonnées de configuration de checkpoint. Chacune est rapportée avec sa
valeur historique et sa valeur de recherche. Tous les états, résidus,
multiplicateurs, itérations et compteurs de calcul restent comparés exactement.
Les temps muraux/CPU et les ressources de chaque exécution sont conservés.

Le contrôleur utilise le verrou partagé du kit et 22 créneaux continus ;
aucun second groupe de 22 processus ne peut s'exécuter simultanément.
Le suivi écrit STATUS.json et une ligne compacte toutes les 10 secondes.
Une interruption, une limite de budget, un reçu absent ou une différence
empêche le passage. Aucun rééchantillonnage automatique n'est effectué.
Un résultat numérique non strict achevé peut servir à la comparaison des
trajectoires, sans être promu en certificat KKT.

Le reçu final est GATE.json, lié par GATE.sha256. La production n'est autorisée
que si passed et production_authorized valent true après les 306 exécutions.
Les tests du contrôleur et les essais locaux ne constituent pas ce reçu.
