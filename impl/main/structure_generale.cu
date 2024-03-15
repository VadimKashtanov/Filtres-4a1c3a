#include "main.cuh"

void ecrire_structure_generale(char * file) {
	FILE * fp = fopen(file, "wb");
	//
	uint constantes[16] = {
		DEPART,
		P,
		P_INTERV,
		N,
		MAX_INTERVALLE,
		SOURCES,
		MAX_PARAMS, NATURES,
		MAX_EMA, MAX_PLUS, MAX_COEF_MACD,
		C, MAX_Y, BLOQUES, F_PAR_BLOQUES,
		INSTS
	};
	//
	FWRITE(constantes, sizeof(uint), 16, fp);
	//
	FWRITE(min_param,  sizeof(uint), NATURES*MAX_PARAMS, fp);
	FWRITE(max_param,  sizeof(uint), NATURES*MAX_PARAMS, fp);
	//
	FWRITE(NATURE_PARAMS,  sizeof(uint), NATURES, fp);
	//
	fclose(fp);
};