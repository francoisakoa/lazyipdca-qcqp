NOYAU DE L'ABLATION DU RAFFINEMENT

Les quatre cellules principales utilisent le même noyau hybride, la même
construction certifiée initiale, l'objectif exact et les égalités signées.
Elles ne diffèrent que par --factorial-refinement frozen|adaptive et par
--factorial-globalization r3|hs. Le backend est un bloc expérimental :
MA27 ou MUMPS+METIS, sans changement de source numérique entre ces blocs.

--factorial-common-guards r3 impose common_r3_guards_v1. Les éligibilités
structurelles, bornes impliquées, démarrages et gardes de stagnation ne sont
plus sélectionnées par la colonne G. Les huit paramètres de globalisation
conservent leurs règles G. La politique commune garde le lift QCQP original :
la réduction native en fonctions de carré signé est désactivée uniquement
pour l'ablation principale, identiquement dans ses quatre cellules. Les
autres réductions quadratiques/affines exactes restent déclarées et comptées.

Le digest guard_config_sha256 lie la politique numérique déclarée pour la
tâche complète, avant consommation des indicateurs de démarrage/presolve par
les phases natives. Les huit paramètres G et le sélecteur frozen/adaptive
sont exclus de ce digest. Il ne prétend pas que toutes les phases appliquent
exactement les mêmes opérations après divergence des trajectoires : les
activations induites par les états restent des conséquences de l'algorithme.

Le témoin frozen bloque les événements eager, périodiques, de rejet et de
récupération de modèle, à leur entrée commune. Tout commit/rollback de métrique
qui atteindrait malgré cela le témoin est refusé. La matrice mathématique de
correction est hashée à la construction et en fin de vie du modèle ; les
modifications adaptatives sont hashées lors des commits et rollbacks. Les
réancrages réutilisent le hash sans relire tous les Hessiennes à chaque pas.
Les caches d'oracle et les compteurs d'activité ne font pas partie de la
matrice mathématique et sont exclus du hash. Le temps du hash est enregistré
dans la rubrique native Diagnostics, et metric_hash_evaluations compte ces
évaluations. Aucun décalage numérique ni seuil d'acceptation n'est relâché.

Les événements metric_initialized portent un index de phase commençant à 1
et initial_inequality_model_sha256. Le premier hash doit coïncider entre
frozen et adaptive à G et backend fixés. La séquence des phases ultérieures
peut diverger après les raffinements ; son hash sert à la traçabilité.
Encodage de initial_hybrid_models_sha256 : pour chaque digest hexadécimal de
64 caractères dans l'ordre des événements, uint64 little-endian de la taille
UTF-8 (64), puis ces 64 octets. SHA256 porte sur cette concaténation ; aucune
fin de ligne ni compteur supplémentaire n'est ajouté.

Le sidecar endpoint.json.factorial.json distingue :
- native_solver_completed : le solveur natif a rendu un endpoint ;
- initial_model_state / initialization_calls : M0 atteint ou non ;
- primary_backend_factor_attempts / primary_newton_backend_used : une tentative
  réelle de factorisation du Newton primaire, et non une simple construction
  de métrique ; cela ne compte pas tous les facteurs auxiliaires/oracles ;
- native_pre_model_return : terminaison avant M0 et avant tout pas accepté.
actual_linear_backend conserve le libellé brut du résultat. Un solveur qui
termine avant M0 n'est pas décrit comme ayant utilisé MA27 ou MUMPS pour le
Newton primaire. La qualification reste fondée sur les trois résidus
originaux et les preuves de données, départ, budget et route.

Les nouveaux sélecteurs absents préservent les chemins des six profils
publiés. La validation des coins historiques et la qualification complète
sur TRX sont néanmoins obligatoires avant production.

Validation locale réalisée par l'agent d'implémentation :
- 155 contrôles C++ des sélecteurs, garde-fous, traces et cas négatifs ;
- 12 contrôles des méthodes réelles d'observation de métrique, compilées dans
  une fixture de workspace typée ; SHA256 recalculé indépendamment en Python ;
- 77 contrôles indépendants des JSON, séquences et terminaisons avant M0.
Ces contrôles ne constituent pas une compilation complète du solveur.
Le contrôle syntaxique de atomic_newton.cpp est bloqué localement par
l'absence de boost/multiprecision/cpp_bin_float.hpp dans les includes par
défaut. Les résultats des builds complets et tests natifs du kit sont consignés
séparément par leur contrôleur. Aucun calcul de campagne n'a été lancé par
l'agent d'implémentation de ce fichier.
