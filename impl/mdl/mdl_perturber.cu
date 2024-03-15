#include "mdl.cuh"

static uint couche_aleatoire(Mdl_t * mdl) {
	uint a = rand() % mdl->total_POIDS;
	FOR(0, i, C) {
		if (a < mdl->inst_POIDS[i]) {
			return i;
		} else {
			a -= mdl->inst_POIDS[i];
		}
	}
	return C-1;
}

//	===================================================

static void perturber_filtre(Mdl_t * mdl) {
	uint f = rand() % (BLOQUES*F_PAR_BLOQUES);
	//
	float r[N];
	r[0] = rnd();
	FOR(1, i, N) r[i] = r[i-1] + rnd()-.5;
	//
	float coef = 0.90;
	FOR(0, i, N) mdl->p[0][f*N + i] = mdl->p[0][f*N + i]*coef + (1-coef)*r[i];
};

void perturber_filtres(Mdl_t * mdl, uint L) {
	FOR(0, i, L) perturber_filtre(mdl);
};

//	===================================================

static void perturber_echanger   (Mdl_t * mdl, uint c) {
	if (mdl->insts[c] == DOT1D) {
		uint X=mdl->Y[c-1], Y=mdl->Y[c];
		uint p0 = (X+1)*(rand()%Y) + (rand()%X);
		uint p1 = (X+1)*(rand()%Y) + (rand()%X);
		float vp0 = mdl->p[c][p0], vp1 = mdl->p[c][p1];
		mdl->p[c][p0] = vp1;
		mdl->p[c][p1] = vp0;
	}
};

static void perturber_diviser  (Mdl_t * mdl, uint c) {
	if (mdl->insts[c] == DOT1D) {
		uint X=mdl->Y[c-1], Y=mdl->Y[c];
		mdl->p[c][(X+1)*(rand()%Y) + (rand()%X)] /= (float)(1+rand()%3);
	}
};

static void perturber_plus_rnd  (Mdl_t * mdl, uint c) {
	if (mdl->insts[c] == DOT1D) {
		uint X=mdl->Y[c-1], Y=mdl->Y[c];
		uint pos = (X+1)*(rand()%Y) + (rand()%X);
		mdl->p[c][pos] += 0.1 * fabs(mdl->p[c][pos]) * 2*(rnd()-0.5);
	}
};

//	======================================

void perturber(Mdl_t * mdl, uint L) {
	mdl_poids_gpu_vers_cpu(mdl);
	FOR(0, i, L) {
		uint c_alea = couche_aleatoire(mdl);
		if (c_alea != 0) {
			float ch = rnd();
			if (0.00 <= ch && ch <= 0.01) perturber_echanger(mdl, c_alea);	// 1%
			if (0.01 <= ch && ch <= 0.10) perturber_diviser (mdl, c_alea);	// 9%
			if (0.10 <= ch && ch <= 1.00) perturber_plus_rnd(mdl, c_alea);	//90%
		} else {
			perturber_filtre(mdl);
		}
	}
	mdl_poids_cpu_vers_gpu(mdl);
	//
	mdl_normer_les_filtres(mdl);
};