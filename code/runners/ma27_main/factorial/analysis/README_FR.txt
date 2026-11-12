Plan factoriel natif — analyses préenregistrées

Avant les trajectoires de qualification, VERIFIER_COUPLAGE.py inspecte les
1 834 QCQP I/M avec les classes C++ natives, y compris les formes issues des
réductions exactes et de la présolution de rang. Aucun problème numérique n'est
résolu pendant ce contrôle. Une réduction signée, une forme potentiellement
concernée ou un contrôle incomplet interdit le lancement factoriel. Le reçu
COUPLING_GATE.json lie chaque inspection aux octets gelés, au départ brut,
au code compilé et au préenregistrement. La relance Ipopt complète reste
indépendante de ce contrôle. Un reçu favorable doit encore être suivi de la
qualification des trajectoires sur les 50 instances préenregistrées.

Les quatre cellules sont R3 (cache, contrôles R3), X1 (cache, contrôles H/S),
X2 (hybride, contrôles R3), H0 (hybride, contrôles H/S). L'objectif et les
égalités originales restent exacts. H0 est distinct de HH, dont la correction
d'objectif est active sur certains problèmes. Le contrôle HH historique sert
à vérifier la rétrocompatibilité et n'entre pas dans les quatre cellules.

Créer le préenregistrement une seule fois, avant les nouveaux résultats :
  python3 PRE_ENREGISTRER.py --contract CONTRAT_4078.json.gz --out PRE_ENREGISTREMENT.json
Le fichier existant ne peut pas être écrasé par le générateur. Son SHA256 est
inscrit dans le manifeste de lancement et chaque observation exportée.

Analyser ensuite :
  python3 ANALYSER_FACTORIEL.py --contract CONTRAT_4078.json.gz \
    --preregistration PRE_ENREGISTREMENT.json --observations FACTORIEL.jsonl --out ANALYSE

FACTORIEL.jsonl est un export du nouveau run, une ligne par instance/cellule.
Champs exigés pour une observation classée : case_id, cell, run_id,
input_sha256, start_sha256, preregistration_sha256, objective_exact=true,
budget_verified=true, trace_success=true,
strict_qcqp (true/false/null), primal, dual, complementarite sous la clé
complementarity. Les champs objective et seconds_process servent aux mesures
secondaires. Un ensemble de plusieurs run_id est refusé : les données
historiques ne peuvent pas remplacer des coins contemporains.

L'analyse retient les 1 834 instances I/M planifiées (1 037 + 797). Les cellules
manquantes ou non vérifiables restent indéterminées et ne deviennent pas des
échecs. Les colonnes de résultats sont donc des bornes lorsque le plan est
incomplet. Les effets principaux sont les différences moyennes sur les deux
modalités de l'autre facteur ; l'interaction est une différence de différences.

La règle confirmatoire porte sur H0 moins X1, dans I ∪ M : au moins 20
certificats supplémentaires ; au moins deux discordances gagnantes pour une
perdante ; aucune collection ne contribue plus de 50 % du gain NET total ;
différence de taux strictement positive avec poids égaux par famille déclarée.
Une conclusion positive exige que toutes les conditions tiennent pour toute
complétion binaire compatible avec les indéterminés. Sinon le résultat est
négatif lorsque les données sont complètes ou qu'une borne exclut la règle ;
il est inconclusif lorsque les inconnus peuvent encore changer la décision.
Les résumés séparés I et M sont descriptifs ; la décision confirmatoire utilise
le groupe I ∪ M prédéfini. Le ratio 0/0 est indéfini, jamais une victoire.
Le rapport détaille les quatre cellules et tous les contrastes par dataset et
par famille déclarée. Les objectifs et temps appariés sont aussi exportés par
ces deux partitions, avec leurs dénominateurs. Les moyennes à poids égaux sont
calculées sur les taux de chaque strate, sans remplacer l'analyse appariée.

Le critère de concentration emploie chaque contribution NETTE de collection,
divisée par le gain NET total positif. Il ne normalise pas uniquement par la
somme des contributions positives. Les bornes préservent le fait que la même
collection apparaît au numérateur et dans le total.

Le champ facultatif accepted_direction_origins est un dictionnaire de comptes
au point final d'acceptation. Les pas avec SOC, rétraction ou autre mélange
restent des classes séparées ; ils ne sont pas appelés des pas majorants purs.
L'export agrège final_origin de la trace, jamais direction_origin seul.
Une absence de provenance est un manque, pas un compteur nul. Le rapport donne
la fraction connue et la couverture de l'instrumentation séparément.

La relance Ipopt est indépendante : les deux backends portent sur la totalité
des 4 078 instances (8 156 tâches), avec réglages numériques par défaut et
qualification KKT originale. Les deux programmes partagent les mêmes 22
créneaux, ou sont exécutés successivement ; on ne lance pas deux groupes de
22 tâches simultanément sur les mêmes cœurs.
