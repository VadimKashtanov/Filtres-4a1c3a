#include "mdl.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

float mdl_les_gains(Mdl_t * mdl, uint t0, uint t1, uint mode, float GRAND_COEF) {
	ASSERT(GRAND_COEF >= 2);
	ASSERT(mdl->T == (t1-t0));
	float * _y = gpu_vers_cpu<float>(mdl->y__d[C-1], (t1-t0)*P);
	float somme = 0;
	float potentiel = 0;
	FOR(t0, t, t1) {
		somme     += powf(fabs(prixs[t+1]/prixs[t]-1),GRAND_COEF) * (signe((prixs[t+1]/prixs[t]-1)) == signe(_y[(t-t0)*P+0]));
		potentiel += powf(fabs(prixs[t+1]/prixs[t]-1),GRAND_COEF);
	}
	free(_y);
	return somme / potentiel;
};

float mdl_score(Mdl_t * mdl, uint t0, uint t1, uint mode) {
	ASSERT(mdl->T == (t1-t0));
	if (mode == 0) mdl_zero_cpu(mdl);
	else           mdl_zero_gpu(mdl);
	//
	mdl_f(mdl, t0, t1, mode);
	//
	float somme_score;
	if (mode == 0) somme_score =  intel_somme_score(mdl->y[C-1],    t0, (t1-t0));
	else           somme_score = nvidia_somme_score(mdl->y__d[C-1], t0, (t1-t0));
	//
	if (mode == 0) return  intel_score_finale(somme_score, (t1-t0));
	else           return nvidia_score_finale(somme_score, (t1-t0));
};

float* mdl_pred(Mdl_t * mdl, uint t0, uint t1, uint mode) {
	ASSERT(mdl->T == (t1-t0));
	if (mode == 0) mdl_zero_cpu(mdl);
	else           mdl_zero_gpu(mdl);
	//
	mdl_f(mdl, t0, t1, mode);
	if (mode == 0) return  intel_prediction(mdl->y[C-1], t0, (t1-t0));
	else           return nvidia_prediction(mdl->y__d[C-1], t0, (t1-t0));
};

void mdl_aller_retour(Mdl_t * mdl, uint t0, uint t1, uint mode) {
	ASSERT(mdl->T == (t1-t0));
	if (mode == 0) mdl_zero_cpu(mdl);
	else           mdl_zero_gpu(mdl);
	mdl_f(mdl, t0, t1, mode);
	//
	float somme_score;
	if (mode == 0) somme_score =  intel_somme_score(mdl->y[C-1], t0, (t1-t0));
	else           somme_score = nvidia_somme_score(mdl->y__d[C-1], t0, (t1-t0));
	//
	float d_score;
	if (mode == 0) d_score =  d_intel_score_finale(somme_score, (t1-t0));
	else           d_score = d_nvidia_score_finale(somme_score, (t1-t0));
	//
	if (mode == 0)  d_intel_somme_score(d_score, mdl->y[C-1],    mdl->dy[C-1], t0, (t1-t0));
	else           d_nvidia_somme_score(d_score, mdl->y__d[C-1], mdl->dy__d[C-1], t0, (t1-t0));
	mdl_df(mdl, t0, t1, mode);
};