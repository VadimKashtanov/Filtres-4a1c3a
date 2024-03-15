#include "mdl.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

void mdl_f(Mdl_t * mdl, uint t0, uint t1, uint mode) {
	FOR(0, c, C) {
		inst_f[mdl->insts[c]](mdl, c, mode, t0, t1);
	};
};