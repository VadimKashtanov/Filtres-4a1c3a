#pragma once

#define DEBBUG false

#include "etc.cuh"

#define PRIXS 56159 //u += u*f*levier*(p[i+L]/p[i]-1)
#define P 1 		//[Nombre de sorties du Model]
#define P_INTERV 10	//(p[i+1+P*P_INTERV]/p[i]-1)

#define N_FLTR  8
#define N       N_FLTR

#define MAX_DECALE_FUTURE 2*N

#define MAX_INTERVALLE 256

#define MAX_MULTPLE_INTERV_NATURES 15

#define MULTIPLE MAX2(N_FLTR,MAX_MULTPLE_INTERV_NATURES)

#define DEPART (MULTIPLE*MAX_INTERVALLE)
#if (DEBBUG == false)
	#define FIN (PRIXS-P-1)
#else
	#define FIN (DEPART+1-MAX_DECALE_FUTURE)
#endif

#define DEPART_FILTRES (MULTIPLE*MAX_INTERVALLE)

//	--- Sources ---

#define SOURCES 5

extern char * nom_sources[SOURCES];

#define   SRC_PRIXS 0
#define SRC_VOLUMES 1
#define    SRC_HIGH 2
#define     SRC_LOW 3
#define  SRC_MEDIAN 4

//	Sources en CPU
extern float   prixs[PRIXS];	//  prixs.bin
extern float volumes[PRIXS];	// volume.bin
extern float    high[PRIXS];	//   high.bin
extern float     low[PRIXS];	//    low.bin
extern float  median[PRIXS];	//  media.bin

extern float * sources[SOURCES];

//	Sources en GPU
extern float *   prixs__d;	//	nVidia
extern float * volumes__d;	//	nVidia
extern float *    high__d;	//	nVidia
extern float *     low__d;	//	nVidia
extern float *   media__d;	// 	nVidia

extern float * sources__d[SOURCES];

void   charger_les_prixs();
void charger_vram_nvidia();
//
void  liberer_cudamalloc();
//
void charger_tout();
void liberer_tout();

//	---	analyse des sources ---

#define MAX_PARAMS 4
#define    NATURES 6

//	-- Analysateurs --
#define     DIRECT 0
#define       MACD 1
#define    CHIFFRE 2
#define    AWESOME 3
#define POURCENT_R 4
#define        RSI 5

uint * cree_DIRECTE();					//classique
uint * cree_MACD(uint k);				//relative
uint * cree_CHIFFRE(uint chiffre);		//theorique
uint * cree_AWESOME(uint k);			//relative
uint * cree_POURCENT_R(uint interv, uint ema_K_post_r);	//theorique
uint * cree_RSI(uint interv);			//theorique

extern uint nature_multiple_interv[NATURES];

extern uint min_param[NATURES][MAX_PARAMS];
extern uint max_param[NATURES][MAX_PARAMS];

extern uint NATURE_PARAMS[NATURES];

extern char * nom_natures[NATURES];

#define MAX_EMA          1024
#define MAX_PLUS         500
#define MAX_COEF_MACD    200
#define MAX_COEF_AWESOME 200
#define MAX_CHIFFRE      10000

typedef struct {
	//	Intervalle
	uint      K_ema;	//ASSERT(1 <=      ema   <= inf           )
	uint intervalle;	//ASSERT(1 <= intervalle <= MAX_INTERVALLE)

	//	Nature
	uint nature;
	/*	Natures: ema-K, macd-k, chiffre-M, dx, dxdx, dxdxdx
			directe : {}							// Juste le Ema_int
			macd    : {coef }   					// le macd sera ema(9*c)-ema(26*c) sur ema(prixs,k)
			chiffre : {cible}						// Peut importe la cible, mais des chiffres comme 50, 100, 1.000 ... sont bien
	*/
	uint params[MAX_PARAMS];

	//	Valeurs
	float   ema[PRIXS];
	float brute[PRIXS];

	//	Gestion des Normes
#define NORME_CLASSIQUE 0 	//[f NORME] r = [(l[i]-min(l))/(max(l) - min(l))]
#define NORME_THEORIQUE 1 	//[f BORNE] r = [(l[i]-min_t)/(max_t-min_t)]
#define NORME_RELATIVE  2   //[f BORNE] r = [(l[i]--max|l]) / (max|l|--max|l|)] (ca devrait etre entre -1;1, mais en fait ca change rien)
	uint  type_de_norme;
	float min_theorique, max_theorique;

	/*
Norme classique : 
	- Des prixs qui peuvent prendre toutes valeurs
	Ex : les prixs du marchee, les volumes ...

Norme Theorique :
	- Des valeurs bornées.
	Ex : Des pourcentages%, rsi, des 'chiffres' ...

Norme Relative :
	- Des valeurs non bornée mais en pratique limités et ou le signe est important
	Ex : macd, awesome ...

	*/

	/*	Note : dans `normalisee` et `dif_normalisee`
	les intervalles sont deja calculee. Donc tout
	ce qui est avant DEPART n'est pas initialisee (car pas utilisee).
	*/
	uint source;
} ema_int_t;

extern char * nom_type_de_norme[3];

void ema_int_calc_ema(ema_int_t * ema_int);

//	Outils qui composent les natures
void _outil_ema(float * y, float * x, uint K);
void _outil_macd(float * y, float * x, float coef);
void _outil_chiffre(float * y, float * x, float chiffre);
void _outil_awesome(float * y, float * x, float coef);
void _outil_pourcent_r(float * y, float * x, uint interv, uint ema);
void _outil_rsi(float * y, float * x, uint interv);

//	Les natures
void nature0__direct    (ema_int_t * ema_int);
void nature1__macd      (ema_int_t * ema_int);
void nature2__chiffre   (ema_int_t * ema_int);
void nature3__awesome   (ema_int_t * ema_int);
void nature4__pourcent_r(ema_int_t * ema_int);
void nature5__rsi       (ema_int_t * ema_int);

typedef void (*nature_f)(ema_int_t*);
extern nature_f fonctions_nature[NATURES];

//	Mem
ema_int_t * cree_ligne(uint source, uint nature, uint K_ema, uint intervalle, uint params[MAX_PARAMS]);
void     liberer_ligne(ema_int_t * ema_int);

//	IO
ema_int_t * lire_ema_int(FILE * fp);
void      ecrire_ema_int(ema_int_t * ema_int, FILE * fp);

//	Visualisation simple Matplotlib
void visualiser_ema_int(
	uint source,
	uint nature,
	uint K_ema, uint intervalle,
	uint params[MAX_PARAMS]);