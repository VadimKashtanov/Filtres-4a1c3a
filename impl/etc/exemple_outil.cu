#include "marchee.cuh"

#include "../impl_tmpl/tmpl_etc.cu"

static void ecrire_courbe(char * fichier, float * courbe, uint L) {
	FILE * fp = fopen(fichier, "wb");
	//
	//FWRITE(&L, sizeof(uint), 1, fp);
	FWRITE(courbe, sizeof(float), L, fp);
	//
	fclose(fp);
};

void visualiser_ema_int(
	uint source,
	uint nature,
	uint K_ema, uint intervalle,
	uint params[MAX_PARAMS])
{
	ema_int_t * ema_int = cree_ligne(
		source, nature, K_ema, intervalle, params);
	//
	float un_filtre[PRIXS] = {0};
	FOR(0, i, PRIXS) un_filtre[i] = 0;
	uint depart = PRIXS - 1 - intervalle*N_FLTR;
	float s = 0;
	float r[N_FLTR];
	FOR(0, i, N_FLTR) {
		s += rnd()-.5;
		r[i] = s;
	}
	float max=r[0], min=r[0];
	FOR(0, i, N_FLTR) {
		if (r[i] > max) max = r[i];
		if (r[i] < min) min = r[i];
	}
	FOR(0, i, N_FLTR) r[i] = (r[i]-min)/(max-min);
	//
	//
	/*max=0; min=0;
	FOR(0, i, PRIXS) {
		if (ema_int->brute[i] > max) max = ema_int->brute[i];
		if (ema_int->brute[i] < min) min = ema_int->brute[i];
	};*/
	max = ema_int->brute[PRIXS-1]; 
	min = ema_int->brute[PRIXS-1-N_FLTR*intervalle];
	FOR(0, i, N_FLTR-1) {
		FOR(0, j, intervalle) {
			float val = r[i] + (r[i+1]-r[i]) * (j / (float)intervalle);
			un_filtre[depart + i*intervalle + j] = val*(max-min) + min;
		}
	}
	//un_filtre[depart + N_FLTR*intervalle] = r[N_FLTR-1]*(max-min) + min;
	//
	//
	ecrire_courbe("tmp_source.float", sources[source], PRIXS);
	ecrire_courbe("tmp_ema.float",       ema_int->ema, PRIXS);
	ecrire_courbe("tmp_brute.float",   ema_int->brute, PRIXS);
	ecrire_courbe("tmp_filtre.float",       un_filtre, PRIXS);
	//
	char cmd[100];
	snprintf(cmd, 100, "python3 afficher_exemple_outil.py %i &", PRIXS);
	//
	SYSTEM(cmd);
	//
	liberer_ligne(ema_int);
}