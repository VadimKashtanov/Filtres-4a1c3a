#include "marchee.cuh"

#include "../impl_tmpl/tmpl_etc.cu"

uint * cree_CHIFFRE(uint chiffre) {
	uint * ret = alloc<uint>(MAX_PARAMS);
	ret[0] = chiffre;
	return ret;
};

void nature2__chiffre(ema_int_t * ema_int) {
	//			-- Parametres --
	uint cible = ema_int->params[0];
	//			-- Assertions --
	ASSERT(min_param[CHIFFRE][0] <= cible && cible <= max_param[CHIFFRE][0]);
	//	-- Transformation des Parametres --
	float chiffre = (float)cible;
	ema_int->type_de_norme = NORME_THEORIQUE;
	ema_int->min_theorique = -1.0;//0.0;
	ema_int->max_theorique = +1.0;//chiffre / 2.0;
	//ema_int->type_de_norme = NORME_CLASSIQUE;
	//		-- Calcule de la Nature --
	_outil_chiffre(ema_int->brute, ema_int->ema, chiffre);
};