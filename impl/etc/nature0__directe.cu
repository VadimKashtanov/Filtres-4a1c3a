#include "marchee.cuh"

#include "../impl_tmpl/tmpl_etc.cu"


uint * cree_DIRECTE() {
	return alloc<uint>(MAX_PARAMS);
}

void nature0__direct (ema_int_t * ema_int) {
	//			-- Parametres --
	//			-- Assertions --
	//	-- Transformation des Parametres --
	ema_int->type_de_norme = NORME_CLASSIQUE;
	//		-- Calcule de la Nature --
	//FOR(0, i, FIN) ema_int->brute[i] = ema_int->ema[i];
	_outil_ema(ema_int->brute, ema_int->ema, 1.0);
};