#include "main.cuh"

#include "../impl_tmpl/tmpl_etc.cu"

uint MODE_t_MODE = t_CONTINUE;
uint grain_t_MODE = 0;

static void visualiser() {
	uint source     = SRC_PRIXS;
	uint nature     = POURCENT_R;
	uint K_ema      = 64;
	uint intervalle = 256;
	uint * params   = cree_POURCENT_R(256, 2);
	visualiser_ema_int(
		source,
		nature,
		K_ema, intervalle,
		params);
};

static void plume_pred(Mdl_t * mdl, uint t0, uint t1) {
	uint fois = (t1-t0)/mdl->T;
	//
	float moyenne_pred[P] = {0};
	float moyenne_les_gain__2 = 0; float coef__2 = 2.0;
	float moyenne_les_gain__4 = 0; float coef__4 = 4.0;
	float moyenne_les_gain__8 = 0; float coef__8 = 8.0;
	//
	FOR(0, i, fois) {
		float * ancien = mdl_pred(mdl, t0 + i*mdl->T, t0 + (i+1)*mdl->T, 3);
		FOR(0, p, P) moyenne_pred[p] += ancien[p];
		free(ancien);
		//
		moyenne_les_gain__2 += mdl_les_gains(mdl, t0 + i*mdl->T, t0 + (i+1)*mdl->T, 3, coef__2);
		moyenne_les_gain__4 += mdl_les_gains(mdl, t0 + i*mdl->T, t0 + (i+1)*mdl->T, 3, coef__4);
		moyenne_les_gain__8 += mdl_les_gains(mdl, t0 + i*mdl->T, t0 + (i+1)*mdl->T, 3, coef__8);
	}
	printf("PRED GENERALE = ");
	FOR(0, p, P) printf(" %f%% ", 100*moyenne_pred[p]/(float)fois);
	printf("  | LES GAINS^2 = %f%% | LES GAINS^4 = %f%% | LES GAINS^8 = %f%%",
		100*moyenne_les_gain__2/(float)fois,
		100*moyenne_les_gain__4/(float)fois,
		100*moyenne_les_gain__8/(float)fois
	);
	printf("\n");
};

float pourcent_masque_nulle[C] = {0};

float * pourcent_masque = de_a(0.40, 0.10, C);

float * alpha = de_a(1e-5, 1e-5, C);

uint * optimiser_tous_les = UNIFORME_C(1);

#define GRAND_T (16*16*10)

#define EMAISATION 10

PAS_OPTIMISER()
int main(int argc, char ** argv) {
	alpha[0] = 1e-5;

	/*	!!!!!!!!!!!!!!!!!!!!!!!!!!

			
			Ajouter l'option d'aleatoire des t avec le pseudo-alea


		!!!!!!!!!!!!!!!!!!!!!!!!!!  */

	//pourcent_masque[0] = 0.70;
	//pourcent_masque[1] = 0.01;

	//	----- Lien constants ------

	//	Faire des instructions cascades et de regard de reflexion (les couches)
	//	sans passer le gradient, donc x constant

	//pourcent_masque_nulle[0] = 0.30;
	//pourcent_masque_nulle[1] = 0.70;

	/*
		* VAPP
	*/
	
	//	-- Init --
	srand(0);
	cudaSetDevice(0);
	titre(" Charger tout ");   charger_tout();

	//	-- Verification --
	titre("Verifier MDL");     verif_mdl_1e5();

	//===============
	titre("  Programme Generale  ");
	ecrire_structure_generale("structure_generale.bin");

	//visualiser();

	/*uint Y[C];
	uint insts[C];
	uint cible[C];
	uint decale_future[C];
	//
	uint st[C][4] = {
		//{4096, DOT1D,          CIBLE_FILTRES_FUTURES,   1 },
		//
	//	  Y      inst             cible             decale-future
		{2048, FILTRES_PRIXS_TOLERANT, CIBLE_NORMALE, NULL},
		{16,   DOT1D,                  CIBLE_NORMALE, NULL},
		//
		{64,   DOT1D,                  CIBLE_NORMALE, NULL},
		{64,   DOT1D,                  CIBLE_NORMALE, NULL},
		{64,   DOT1D,                  CIBLE_NORMALE, NULL},
		{64,   DOT1D,                  CIBLE_NORMALE, NULL},
		{64,   DOT1D,                  CIBLE_NORMALE, NULL},
		{64,   DOT1D,                  CIBLE_NORMALE, NULL},
		{64,   DOT1D,                  CIBLE_NORMALE, NULL},
		{64,   DOT1D,                  CIBLE_NORMALE, NULL},
		{64,   DOT1D,                  CIBLE_NORMALE, NULL},
		{64,   DOT1D,                  CIBLE_NORMALE, NULL},
		{64,   DOT1D,                  CIBLE_NORMALE, NULL},
		{64,   DOT1D,                  CIBLE_NORMALE, NULL},
		{64,   DOT1D,                  CIBLE_NORMALE, NULL},
		{64,   DOT1D,                  CIBLE_NORMALE, NULL},
		{64,   DOT1D,                  CIBLE_NORMALE, NULL},
		{64,   DOT1D,                  CIBLE_NORMALE, NULL},
		{64,   DOT1D,                  CIBLE_NORMALE, NULL},
		{64,   DOT1D,                  CIBLE_NORMALE, NULL},
		{64,   DOT1D,                  CIBLE_NORMALE, NULL},
		{64,   DOT1D,                  CIBLE_NORMALE, NULL},
		{64,   DOT1D,                  CIBLE_NORMALE, NULL},
		{64,   DOT1D,                  CIBLE_NORMALE, NULL},
		{64,   DOT1D,                  CIBLE_NORMALE, NULL},
		{64,   DOT1D,                  CIBLE_NORMALE, NULL},
		{64,   DOT1D,                  CIBLE_NORMALE, NULL},
		{64,   DOT1D,                  CIBLE_NORMALE, NULL},
		{64,   DOT1D,                  CIBLE_NORMALE, NULL},
		{64,   DOT1D,                  CIBLE_NORMALE, NULL},
		//
		{P,    DOT1D,                  CIBLE_NORMALE, NULL}
	};
	FOR(0, i, C) {
		    Y[i] = st[i][0];
		insts[i] = st[i][1];
		cible[i] = st[i][2];
		decale_future[i] = st[i][3];
	}
	//
	//	Assurances :
	ema_int_t * bloque[BLOQUES] = {
	//			    Source,      Nature,  K_ema, Intervalle,     {params}
		cree_ligne(SRC_PRIXS, DIRECT, 1, 1, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 1, 2, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 1, 2, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 1, 4, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 2, 1.0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 2, 2, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 2, 4, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 2, 8, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 4, 2.0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 4, 4, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 4, 8, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 4, 16, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 8, 4.0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 8, 8, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 8, 16, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 8, 32, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 16, 8.0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 16, 8.0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 16, 16, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 16, 32, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 16, 64, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 32, 16.0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 32, 32, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 32, 64, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 32, 128, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 64, 32.0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 64, 64, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 64, 128, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 64, 256, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 128, 64.0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 128, 128, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 128, 256, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 256, 128.0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 256, 256, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 1, 1, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 1, 2, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 1, 2, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 1, 4, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 2, 1.0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 2, 2, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 2, 4, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 2, 8, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 4, 2.0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 4, 4, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 4, 8, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 4, 16, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 8, 4.0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 8, 8, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 8, 16, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 8, 32, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 16, 8.0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 16, 8.0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 16, 16, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 16, 32, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 16, 64, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 32, 16.0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 32, 32, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 32, 64, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 32, 128, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 64, 32.0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 64, 64, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 64, 128, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 64, 256, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 128, 64.0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 128, 128, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 128, 256, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 256, 128.0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 256, 256, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 1, 1, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 1, 2, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 1, 2, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 1, 4, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 2, 1.0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 2, 2, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 2, 4, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 2, 8, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 4, 2.0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 4, 4, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 4, 8, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 4, 16, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 8, 4.0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 8, 8, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 8, 16, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 8, 32, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 16, 8.0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 16, 8.0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 16, 16, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 16, 32, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 16, 64, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 32, 16.0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 32, 32, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 32, 64, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 32, 128, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 64, 32.0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 64, 64, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 64, 128, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 64, 256, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 128, 64.0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 128, 128, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 128, 256, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 256, 128.0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 256, 256, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 1, 1, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 1, 2, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 1, 2, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 1, 4, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 2, 1.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 2, 2, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 2, 4, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 2, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 4, 2.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 4, 4, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 4, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 4, 16, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 8, 4.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 8, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 8, 16, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 8, 32, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 16, 8.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 16, 8.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 16, 16, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 16, 32, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 16, 64, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 32, 16.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 32, 32, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 32, 64, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 32, 128, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 64, 32.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 64, 64, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 64, 128, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 64, 256, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 128, 64.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 128, 128, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 128, 256, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 256, 128.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 256, 256, cree_DIRECTE()),
	// -------
		cree_ligne(SRC_PRIXS, MACD, 1, 8, cree_MACD(1)),
		cree_ligne(SRC_PRIXS, MACD, 1, 8, cree_MACD(4)),
		cree_ligne(SRC_PRIXS, MACD, 2, 2, cree_MACD(1)),
		cree_ligne(SRC_PRIXS, MACD, 2, 16, cree_MACD(2)),
		cree_ligne(SRC_PRIXS, MACD, 2, 16, cree_MACD(8)),
		cree_ligne(SRC_PRIXS, MACD, 4, 2.0, cree_MACD(1)),
		cree_ligne(SRC_PRIXS, MACD, 4, 4, cree_MACD(2)),
		cree_ligne(SRC_PRIXS, MACD, 4, 32, cree_MACD(4)),
		cree_ligne(SRC_PRIXS, MACD, 4, 32, cree_MACD(16)),
		cree_ligne(SRC_PRIXS, MACD, 8, 4.0, cree_MACD(2)),
		cree_ligne(SRC_PRIXS, MACD, 8, 8, cree_MACD(1)),
		cree_ligne(SRC_PRIXS, MACD, 8, 64, cree_MACD(8)),
		cree_ligne(SRC_PRIXS, MACD, 8, 64, cree_MACD(32)),
		cree_ligne(SRC_PRIXS, MACD, 16, 8.0, cree_MACD(1)),
		cree_ligne(SRC_PRIXS, MACD, 16, 8.0, cree_MACD(4)),
		cree_ligne(SRC_PRIXS, MACD, 16, 16, cree_MACD(8)),
		cree_ligne(SRC_PRIXS, MACD, 16, 128, cree_MACD(16)),
		cree_ligne(SRC_PRIXS, MACD, 16, 128, cree_MACD(64)),
		cree_ligne(SRC_PRIXS, MACD, 32, 16.0, cree_MACD(2)),
		cree_ligne(SRC_PRIXS, MACD, 32, 16.0, cree_MACD(8)),
		cree_ligne(SRC_PRIXS, MACD, 32, 32, cree_MACD(4)),
		cree_ligne(SRC_PRIXS, MACD, 32, 32, cree_MACD(16)),
		cree_ligne(SRC_PRIXS, MACD, 32, 256, cree_MACD(128)),
		cree_ligne(SRC_PRIXS, MACD, 64, 32.0, cree_MACD(4)),
		cree_ligne(SRC_PRIXS, MACD, 64, 32.0, cree_MACD(16)),
		cree_ligne(SRC_PRIXS, MACD, 64, 64, cree_MACD(8)),
		cree_ligne(SRC_PRIXS, MACD, 256, 128.0, cree_MACD(16)),
		cree_ligne(SRC_PRIXS, MACD, 256, 128.0, cree_MACD(64)),
		cree_ligne(SRC_PRIXS, MACD, 256, 256, cree_MACD(32)),
		cree_ligne(SRC_PRIXS, MACD, 256, 256, cree_MACD(128)),
		cree_ligne(SRC_HIGH, MACD, 1, 8, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 1, 8, cree_MACD(4)),
		cree_ligne(SRC_HIGH, MACD, 2, 2, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 2, 16, cree_MACD(2)),
		cree_ligne(SRC_HIGH, MACD, 2, 16, cree_MACD(8)),
		cree_ligne(SRC_HIGH, MACD, 4, 2.0, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 4, 4, cree_MACD(2)),
		cree_ligne(SRC_HIGH, MACD, 4, 32, cree_MACD(4)),
		cree_ligne(SRC_HIGH, MACD, 4, 32, cree_MACD(16)),
		cree_ligne(SRC_HIGH, MACD, 8, 4.0, cree_MACD(2)),
		cree_ligne(SRC_HIGH, MACD, 8, 8, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 8, 64, cree_MACD(8)),
		cree_ligne(SRC_HIGH, MACD, 8, 64, cree_MACD(32)),
		cree_ligne(SRC_HIGH, MACD, 16, 8.0, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 16, 8.0, cree_MACD(4)),
		cree_ligne(SRC_HIGH, MACD, 16, 16, cree_MACD(8)),
		cree_ligne(SRC_HIGH, MACD, 16, 128, cree_MACD(16)),
		cree_ligne(SRC_HIGH, MACD, 16, 128, cree_MACD(64)),
		cree_ligne(SRC_HIGH, MACD, 32, 16.0, cree_MACD(2)),
		cree_ligne(SRC_HIGH, MACD, 32, 16.0, cree_MACD(8)),
		cree_ligne(SRC_HIGH, MACD, 32, 32, cree_MACD(4)),
		cree_ligne(SRC_HIGH, MACD, 32, 32, cree_MACD(16)),
		cree_ligne(SRC_HIGH, MACD, 32, 256, cree_MACD(128)),
		cree_ligne(SRC_HIGH, MACD, 64, 32.0, cree_MACD(4)),
		cree_ligne(SRC_HIGH, MACD, 64, 32.0, cree_MACD(16)),
		cree_ligne(SRC_HIGH, MACD, 64, 64, cree_MACD(8)),
		cree_ligne(SRC_HIGH, MACD, 256, 128.0, cree_MACD(16)),
		cree_ligne(SRC_HIGH, MACD, 256, 128.0, cree_MACD(64)),
		cree_ligne(SRC_HIGH, MACD, 256, 256, cree_MACD(32)),
		cree_ligne(SRC_HIGH, MACD, 256, 256, cree_MACD(128)),
		cree_ligne(SRC_LOW, MACD, 1, 8, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 1, 8, cree_MACD(4)),
		cree_ligne(SRC_LOW, MACD, 2, 2, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 2, 16, cree_MACD(2)),
		cree_ligne(SRC_LOW, MACD, 2, 16, cree_MACD(8)),
		cree_ligne(SRC_LOW, MACD, 4, 2.0, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 4, 4, cree_MACD(2)),
		cree_ligne(SRC_LOW, MACD, 4, 32, cree_MACD(4)),
		cree_ligne(SRC_LOW, MACD, 4, 32, cree_MACD(16)),
		cree_ligne(SRC_LOW, MACD, 8, 4.0, cree_MACD(2)),
		cree_ligne(SRC_LOW, MACD, 8, 8, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 8, 64, cree_MACD(8)),
		cree_ligne(SRC_LOW, MACD, 8, 64, cree_MACD(32)),
		cree_ligne(SRC_LOW, MACD, 16, 8.0, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 16, 8.0, cree_MACD(4)),
		cree_ligne(SRC_LOW, MACD, 16, 16, cree_MACD(8)),
		cree_ligne(SRC_LOW, MACD, 16, 128, cree_MACD(16)),
		cree_ligne(SRC_LOW, MACD, 16, 128, cree_MACD(64)),
		cree_ligne(SRC_LOW, MACD, 32, 16.0, cree_MACD(2)),
		cree_ligne(SRC_LOW, MACD, 32, 16.0, cree_MACD(8)),
		cree_ligne(SRC_LOW, MACD, 32, 32, cree_MACD(4)),
		cree_ligne(SRC_LOW, MACD, 32, 32, cree_MACD(16)),
		cree_ligne(SRC_LOW, MACD, 32, 256, cree_MACD(128)),
		cree_ligne(SRC_LOW, MACD, 64, 32.0, cree_MACD(4)),
		cree_ligne(SRC_LOW, MACD, 64, 32.0, cree_MACD(16)),
		cree_ligne(SRC_LOW, MACD, 64, 64, cree_MACD(8)),
		cree_ligne(SRC_LOW, MACD, 256, 128.0, cree_MACD(16)),
		cree_ligne(SRC_LOW, MACD, 256, 128.0, cree_MACD(64)),
		cree_ligne(SRC_LOW, MACD, 256, 256, cree_MACD(32)),
		cree_ligne(SRC_LOW, MACD, 256, 256, cree_MACD(128)),
		cree_ligne(SRC_VOLUMES, MACD, 1, 8, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 1, 8, cree_MACD(4)),
		cree_ligne(SRC_VOLUMES, MACD, 2, 2, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 2, 16, cree_MACD(2)),
		cree_ligne(SRC_VOLUMES, MACD, 2, 16, cree_MACD(8)),
		cree_ligne(SRC_VOLUMES, MACD, 4, 2.0, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 4, 4, cree_MACD(2)),
		cree_ligne(SRC_VOLUMES, MACD, 4, 32, cree_MACD(4)),
		cree_ligne(SRC_VOLUMES, MACD, 4, 32, cree_MACD(16)),
		cree_ligne(SRC_VOLUMES, MACD, 8, 4.0, cree_MACD(2)),
		cree_ligne(SRC_VOLUMES, MACD, 8, 8, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 8, 64, cree_MACD(8)),
		cree_ligne(SRC_VOLUMES, MACD, 8, 64, cree_MACD(32)),
		cree_ligne(SRC_VOLUMES, MACD, 16, 8.0, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 16, 8.0, cree_MACD(4)),
		cree_ligne(SRC_VOLUMES, MACD, 16, 16, cree_MACD(8)),
		cree_ligne(SRC_VOLUMES, MACD, 16, 128, cree_MACD(16)),
		cree_ligne(SRC_VOLUMES, MACD, 16, 128, cree_MACD(64)),
		cree_ligne(SRC_VOLUMES, MACD, 32, 16.0, cree_MACD(2)),
		cree_ligne(SRC_VOLUMES, MACD, 32, 16.0, cree_MACD(8)),
		cree_ligne(SRC_VOLUMES, MACD, 32, 32, cree_MACD(4)),
		cree_ligne(SRC_VOLUMES, MACD, 32, 32, cree_MACD(16)),
		cree_ligne(SRC_VOLUMES, MACD, 32, 256, cree_MACD(128)),
		cree_ligne(SRC_VOLUMES, MACD, 64, 32.0, cree_MACD(4)),
		cree_ligne(SRC_VOLUMES, MACD, 64, 32.0, cree_MACD(16)),
		cree_ligne(SRC_VOLUMES, MACD, 64, 64, cree_MACD(8)),
		cree_ligne(SRC_VOLUMES, MACD, 256, 128.0, cree_MACD(16)),
		cree_ligne(SRC_VOLUMES, MACD, 256, 128.0, cree_MACD(64)),
		cree_ligne(SRC_VOLUMES, MACD, 256, 256, cree_MACD(32)),
		cree_ligne(SRC_VOLUMES, MACD, 256, 256, cree_MACD(128)),
	};
	//
	Mdl_t * mdl = cree_mdl(GRAND_T, Y, insts, cible, decale_future, bloque);*/

	Mdl_t * mdl = ouvrire_mdl(GRAND_T, "mdl.bin");

	//mdl_re_cree_poids(mdl);

	//uint c=5, nouveau_Y=64;
	//mdl_changer_couche_Y(mdl, c, nouveau_Y);

	enregistrer_les_lignes_brute(mdl, "lignes_brute.bin");

	plumer_mdl(mdl);

	//	================= Initialisation ==============
	uint t0 = DEPART;
	uint t1 = t0 + ROND_MODULO((FIN-DEPART), (16*16));
	printf("t0=%i t1=%i FIN=%i (t1-t0=%i, %%(16*16)=%i)\n", t0, t1, FIN, t1-t0, (t1-t0)%(16*16));
	//
	plume_pred(mdl, t0, t1);
	//comportement(mdl, t0, t0+GRAND_T);
	//
	srand(time(NULL));
#define PERTURBATIONS 0
	//
	uint rep = 0;
	while (1) {
		//perturber(mdl, 100);
		//perturber_filtres(mdl, 10);
		//
		optimisation_mini_packet(
			mdl,
			t0, t1, GRAND_T,
			alpha, 1.0,
			/*SGD*//*RMSPROP*/ADAM, 1000,//5000,
			//pourcent_masque,
			pourcent_masque_nulle,
			PERTURBATIONS,
			optimiser_tous_les);
		//
		mdl_poids_gpu_vers_cpu(mdl);
		ecrire_mdl(mdl, "mdl.bin");
		if (rep % 5 == 0) plume_pred(mdl, t0, t1);
		//
		printf("===================================================\n");
		printf("================= TERMINE %i ======================\n", rep++);
		printf("===================================================\n");
	}
	//
	mdl_gpu_vers_cpu(mdl);
	ecrire_mdl(mdl, "mdl.bin");
	liberer_mdl(mdl);

	//	-- Fin --
	liberer_tout();
};