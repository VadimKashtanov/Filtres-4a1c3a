#include "marchee.cuh"

#include "../impl_tmpl/tmpl_etc.cu"

uint * cree_POURCENT_R(uint interv, uint ema_k_post_r) {
	uint * ret = alloc<uint>(MAX_PARAMS);
	ret[0] = interv;
	ret[1] = ema_k_post_r;
	return ret;
};

void nature4__pourcent_r(ema_int_t * ema_int) {
	//			-- Parametres --
	uint interv = ema_int->params[0];
	uint ema_k_post_r = ema_int->params[0];
	//			-- Assertions --
	ASSERT(min_param[POURCENT_R][0] <= interv && interv <= max_param[POURCENT_R][0]);
	ASSERT(min_param[POURCENT_R][1] <= ema_k_post_r && ema_k_post_r <= max_param[POURCENT_R][1]);
	//	-- Transformation des Parametres --
	ema_int->type_de_norme = NORME_THEORIQUE;
	ema_int->min_theorique = -1.0;
	ema_int->max_theorique = +0.0;
	//		-- Calcule de la Nature --
	_outil_pourcent_r(ema_int->brute, ema_int->ema, interv, ema_k_post_r);
};