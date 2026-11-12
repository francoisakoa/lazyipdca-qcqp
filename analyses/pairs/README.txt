LAZYQCQP — APPARIEMENTS IPOPT ALIGNÉ / DOUZE VARIANTES NATIVES — R14

Ce dossier analyse 65 248 observations archivées : 4 078 instances pour
chacune des douze configurations natives (six politiques, MA27/MUMPS) et
quatre configurations Ipopt (monotone/adaptative, MA27/MUMPS).
Il n'exécute aucun optimiseur, ne réaudite aucun endpoint et ne modifie
aucune classification. Les mêmes empreintes QCQP et x0 sont vérifiées
avant les appariements. Le contrôle indépendant des RESULT/AUDIT/options/
budgets de l'archive est livré séparément dans ../audit.

REPRODUCTION — Python 3.12 ; NumPy seulement pour le sous-dossier reference
Depuis ce dossier, ou en remplaçant les chemins par leurs équivalents :
  python3 analyze_aligned_pairs.py --compact OBSERVATIONS_16_CONFIGS.jsonl.gz \
    --native-details NATIVE_R3_MA27_DETAILS.json --out reproduced
  python3 build_full_tables.py --input-directory reproduced --out reproduced
  python3 reference/analyse_reference_values.py \
    --compact reproduced/OBSERVATIONS_16_CONFIGS.jsonl.gz --out reproduced/reference
  python3 reference/build_reference_tables.py \
    --input-directory reproduced/reference --out reproduced/reference
Le compact est copié dans la sortie si nécessaire. Les entrées ne sont pas
modifiées. Les instantanés publiés et les petits contrats de conversion
sont fournis dans reference/data ; aucun accès réseau n'est nécessaire.

MESURES
PAIR_SUMMARIES couvre 90 paires sur sept périmètres : tous les cas, I, M,
E, Q, ASU dans M, et M hors ASU. Les paires incluent toutes les douze
variantes natives face aux quatre références Ipopt, les politiques natives
au sein de chaque backend, les six comparaisons de backend à politique
native fixée, et les six paires entre références Ipopt.

La couverture distingue les quatre catégories appariées et conserve tous
les cas prévus. Les dossiers techniques restent identifiés dans les CSV,
sans les assimiler à des échecs numériques. Les objectifs utilisent leurs
propres cohortes de certificats communs, avec médianes signées et absolues,
quartiles et seuils 10^-3 et 10^-2. Un écart positif favorise B. Les temps
emploient le ratio de temps de processus A/B, sans retrait de démarrage ;
la restriction d'une seconde s'applique aux deux processus. Une sensibilité
combine ce seuil temporel et l'équivalence des objectifs. Les résidus finaux
appariés sont aussi résumés dans PAIR_SUMMARIES.

REFERENCE fournit les écarts aux 49 points publiés QPLIB et quatre points
MINLPLib EMFL précédemment admis par leurs contrats de conversion. Le
périmètre et les limites de ces références sont conservés : aucun optimum
global démontré et aucune nouvelle référence M n'est revendiqué. La cohorte
principale commune compte 50 cas : 46 QPLIB et quatre EMFL, certifiés par
R3-MA27, HH-MA27 et les deux références Ipopt monotones. Les autres
configurations conservent leur propre nombre de certificats sur ces 50 cas.

COUPLES PRIORITAIRES I
R3_IPOPT_REGIME_I_CONFIRMED_PAIRS.csv contient les 128 croisements archivés
R3 non strict / Ipopt strict, sur 32 identités, avec les deux backends
natifs et les quatre références Ipopt. SAME_BACKEND contient les 64
croisements qui gardent le backend. Les quatre croisements où R3 est
indéterminé sont séparés. Les données/départs sont appariés et les limites
nominales sont déclarées ; ce tableau n'est pas une nouvelle reproduction
des trajectoires. Les itérations et compteurs R3-MA27 sont enrichis à
partir d'observations natives authentifiées par les mêmes empreintes.
Les valeurs MUMPS absentes restent vides ; le message natif détaillé et
les traces nécessaires à une cause algorithmique ne sont pas inventés.

VALIDATION
Les 48 936 lignes natives ont été comparées aux projections R13 : mêmes
classifications, objectifs, résidus, statuts, données, départs, profils,
binaires et temps. Le reçu INDEPENDENT_MANUSCRIPT_TABLE_CHECK vérifie 864
cellules numériques des tableaux R14 calculés indépendamment.
Le reçu ARCHIVED_PRODUCER_SUMMARY_CHECK vérifie aussi 11 340 cellules
numériques des résumés livrés dans l'archive : les comptages sont identiques
et le plus grand écart flottant est 1,43e-14. Le test portable reproduit
19 tableaux et les 630 résumés appariés à partir d'une copie isolée.
Les résultats sont descriptifs de ces corpus et configurations complètes ;
les populations des cohortes appariées peuvent différer entre comparaisons.
