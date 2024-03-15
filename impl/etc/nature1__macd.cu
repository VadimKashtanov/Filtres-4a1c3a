#include "marchee.cuh"

#include "../impl_tmpl/tmpl_etc.cu"

uint * cree_MACD(uint k) {
	uint * ret = alloc<uint>(MAX_PARAMS);
	ret[0] = k;
	return ret;
}

void nature1__macd(ema_int_t * ema_int) {
	//			-- Parametres --
	uint coef = ema_int->params[0];
	//			-- Assertions --
	ASSERT(min_param[MACD][0] <= coef && coef <= max_param[MACD][0]);
	//	-- Transformation des Parametres --
	float _coef = coef;
	ema_int->type_de_norme = /*NORME_CLASSIQUE;//*/NORME_RELATIVE;
	//		-- Calcule de la Nature --
	_outil_macd(ema_int->brute, ema_int->ema, _coef);
};