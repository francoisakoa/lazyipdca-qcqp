# Audit du retrait des bornes : R3 et huit cellules de raffinement

Lecture seule, sans optimisation ni accès à TRX. Les deux sources historiques
ont été recontrôlées : 823 fichiers sur 823 correspondent à chacun des
manifestes MA27 et MUMPS.

Le retrait des côtés de bornes impliqués par une **égalité quadratique
radiale diagonale** est activé dans les sources historiques seulement pour
HH, HS et SH (et le mode interne H0), sous le moteur majorants et le booléen
maître `dca_rank_implied_bounds`. R3 utilise `convexification_study=off`,
qui exclut cette branche, comme SS. Référence :
`src/rank_presolve.cpp:925–931`, identique dans les deux sources.

Les 503 observations ASU R3 MA27 confirment zéro côté retiré par ce
prétraitement, aucune réduction de dimension et le départ préservé. Les
501 modèles à 64 variables ont 32 égalités quadratiques ; les deux modèles
`beam__asu_u1_rf2` et `beam__asu_u2_rf2` ont 32 variables et 16 égalités.
Leurs 128 ou 64 côtés de boîtes expliquent la différence de compteurs de
HH/HS/SH. Le jeu projeté MUMPS ne contient pas ces compteurs détaillés :
aucun second constat empirique de 503 compteurs MUMPS n'est revendiqué.

Dans les huit cellules configurées de raffinement, le profil de base HH
est complété par `factorial-common-guards=r3`. Le prédicat
`factorial_hs_guards()` est donc faux, indépendamment du facteur de
globalisation ; le gate de retrait radial par égalité quadratique est
désactivé dans les huit cellules. Le prédicat `factorial_r3_controls()` est
vrai : le gate de retrait des bornes impliquées par une **égalité affine
singleton** reste disponible. Il conserve l'égalité et la variable ; ce
n'est pas une élimination de singleton. Références :
`runtime/PROFILES_EXPERIENCE.py:11–12,34–44`,
`native/experiment_source/include/lazyipdca_qcqp/settings.hpp:33–49`,
`native/experiment_source/src/rank_presolve.cpp:925–932`.
Ces constats portent sur les sources et profils locaux, pas sur les
résultats ou l'état d'une campagne distante.

L'option actuelle `--dca-rank-implied-bounds 0|1` est un maître commun :
elle ne permet pas d'activer seulement le retrait radial par égalité pour
R3 ou SS. Une intervention propre demande un contrôle indépendant
`profile|keep|remove` ne modifiant que ce gate. La valeur `profile` doit
reproduire l'historique ; `remove` active la preuve exacte sans changer
les corrections, contrôleurs ou autres réductions. Ce correctif n'a pas
été appliqué par cet audit.

Le rapport compagnon `OTHER_BOUND_PATHS.json` répertorie douze chemins
connexes. La substitution à facteur commun transforme les bornes en
inégalités après un changement de variables ; elle est observée inactive
sur les 503 R3 ASU. Les promotions de bornes du Newton atomique changent
seulement la représentation condensée ou bordée de la même matrice KKT.
Elles ne retirent aucun côté ni terme barrière. Les autres réductions
exactes restent distinguées ; les compteurs du rank presolve ne suffisent
pas à exclure toutes leurs inspections.

La preuve radiale implique la redondance sur le domaine faisable original.
Elle ne dit pas si les côtés sont stricts simultanément ou forcés actifs.
Par exemple, un cercle possède des points strictement à l'intérieur de sa
boîte, tandis qu'une égalité quadratique singleton peut imposer un bord.
Sans les coefficients ASU gelés et les traces détaillées, cet audit ne
qualifie pas cet intérieur et n'attribue aucune fraction des écarts
d'objectif aux termes barrières supprimés.
