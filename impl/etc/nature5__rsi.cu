#include "marchee.cuh"

#include "../impl_tmpl/tmpl_etc.cu"

uint * cree_RSI(uint interv) {
	uint * ret = alloc<uint>(MAX_PARAMS);
	ret[0] = interv;
	return ret;
};

void nature5__rsi(ema_int_t * ema_int) {
	//			-- Parametres --
	uint interv = ema_int->params[0];
	//			-- Assertions --
	ASSERT(min_param[RSI][0] <= interv && interv <= max_param[RSI][0]);
	//	-- Transformation des Parametres --
	float _coef = interv;
	ema_int->type_de_norme = NORME_THEORIQUE;
	ema_int->min_theorique = 0.0;
	ema_int->max_theorique = 1.0;
	//		-- Calcule de la Nature --
	_outil_rsi(ema_int->brute, ema_int->ema, interv);
};