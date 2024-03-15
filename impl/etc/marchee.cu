#include "mdl.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

/*
def lire_ligne_model(ligne):
	with open("/home/vadim/Bureau/Filtres-V1.4+ (versions)/3b1a/lignes_brute.bin", "rb") as co: bins = co.read()
	import struct as st
	import matplotlib.pyplot as plt
	BLOQUES, PRIXS = st.unpack('II', bins[:8])
	__lignes = [st.unpack('f'*PRIXS, bins[8+(i*PRIXS)*4:8+4*(i+1)*PRIXS]) for i in range(BLOQUES)]
	plt.plot(__lignes[ligne]);plt.show()
*/

char * nom_sources[SOURCES] = {
	" prixs ",
	"volumes",
	"  haut ",
	"  bas  ",
	" median"
};

//	Sources
float   prixs[PRIXS] = {};
float volumes[PRIXS] = {};
float    high[PRIXS] = {};
float     low[PRIXS] = {};
float  median[PRIXS] = {};

float *   prixs__d = 0x0;
float * volumes__d = 0x0;
float *    high__d = 0x0;
float *     low__d = 0x0;
float *  median__d = 0x0;

float * sources[SOURCES] = {
	prixs, volumes, high, low, median
};

float * sources__d[SOURCES] = {
	prixs__d, volumes__d, high__d, low__d, median__d
};

void charger_les_prixs() {
	uint __PRIXS;
	FILE * fp;
	//
	fp = fopen("prixs/prixs.bin", "rb");
	ASSERT(fp != 0);
	(void)!fread(&__PRIXS, sizeof(uint), 1, fp);
	ASSERT(__PRIXS == PRIXS);
	(void)!fread(prixs, sizeof(float), PRIXS, fp);
	fclose(fp);
	//
	fp = fopen("prixs/volumes.bin", "rb");
	ASSERT(fp != 0);
	(void)!fread(&__PRIXS, sizeof(uint), 1, fp);
	ASSERT(__PRIXS == PRIXS);
	(void)!fread(volumes, sizeof(float), PRIXS, fp);
	fclose(fp);
	//
	fp = fopen("prixs/high.bin", "rb");
	ASSERT(fp != 0);
	(void)!fread(&__PRIXS, sizeof(uint), 1, fp);
	ASSERT(__PRIXS == PRIXS);
	(void)!fread(high, sizeof(float), PRIXS, fp);
	fclose(fp);
	//
	fp = fopen("prixs/low.bin", "rb");
	ASSERT(fp != 0);
	(void)!fread(&__PRIXS, sizeof(uint), 1, fp);
	ASSERT(__PRIXS == PRIXS);
	(void)!fread(low, sizeof(float), PRIXS, fp);
	fclose(fp);
	//
	fp = fopen("prixs/median.bin", "rb");
	ASSERT(fp != 0);
	(void)!fread(&__PRIXS, sizeof(uint), 1, fp);
	ASSERT(__PRIXS == PRIXS);
	(void)!fread(median, sizeof(float), PRIXS, fp);
	fclose(fp);
};

//	===========================================================

void ema_int_calc_ema(ema_int_t * ema_int) {
	//			-- Parametres --
	uint K = ema_int->K_ema;
	float _K = 1.0 / ((float)K);
	//	EMA
	ema_int->ema[0] = sources[ema_int->source][0];
	FOR(1, i, PRIXS) {
		ema_int->ema[i] = ema_int->ema[i-1] * (1.0 - _K) + sources[ema_int->source][i]*_K;
	}
};

//	===========================================================

uint nature_multiple_interv[NATURES] = {
	0,
	0,
	0,
	14,
	14
};

nature_f fonctions_nature[NATURES] = {
	nature0__direct,
	nature1__macd,
	nature2__chiffre,
	nature3__awesome,
	nature4__pourcent_r,
	nature5__rsi,
};

uint NATURE_PARAMS[NATURES] = {
	0,
	1,
	1,
	1,
	2,
	2
};

uint min_param[NATURES][MAX_PARAMS] = {
	{0,0,0,0},
	{1,0,0,0},
	{1,0,0,0},
	{1,0,0,0},
	{1,1,0,0},
	{1,1,0,0}
};

uint max_param[NATURES][MAX_PARAMS] = {
	{0,                0,       0,        0      }, 
	{MAX_COEF_MACD,    0,       0,        0      },
	{MAX_CHIFFRE,      0,       0,        0      },
	{MAX_COEF_AWESOME, 0,       0,        0      },
	{MAX_INTERVALLE,   MAX_EMA, 0,        0      },
	{MAX_INTERVALLE,   MAX_EMA, 0,        0      } 
};

char * nom_natures[NATURES] {
	"directe",
	"  macd ",
	"chiffre",
	"awesome",
	"  %R   ",
	"  RSI  "
};

ema_int_t * cree_ligne(uint source, uint nature, uint K_ema, uint intervalle, uint params[MAX_PARAMS]) {
	ema_int_t * ret = alloc<ema_int_t>(1);
	//
	ret->source = source;
	ret->nature = nature;
	ret->K_ema  = K_ema;
	ret->intervalle = intervalle;
	//
	ASSERT(intervalle <= MAX_INTERVALLE);
	ASSERT(K_ema      <= MAX_EMA);
	//
	memcpy(ret->params, params, sizeof(uint) * MAX_PARAMS);
	//
	ema_int_calc_ema(ret);
	fonctions_nature[nature](ret);
	//
	return ret;
};

void liberer_ligne(ema_int_t * ema_int) {

};

void charger_vram_nvidia() {
	CONTROLE_CUDA(cudaMalloc((void**)&  prixs__d, sizeof(float) * PRIXS));
	CONTROLE_CUDA(cudaMalloc((void**)&volumes__d, sizeof(float) * PRIXS));
	CONTROLE_CUDA(cudaMalloc((void**)&   high__d, sizeof(float) * PRIXS));
	CONTROLE_CUDA(cudaMalloc((void**)&    low__d, sizeof(float) * PRIXS));
	CONTROLE_CUDA(cudaMalloc((void**)& median__d, sizeof(float) * PRIXS));
	//
	CONTROLE_CUDA(cudaMemcpy(  prixs__d,   prixs, sizeof(float) * PRIXS, cudaMemcpyHostToDevice));
	CONTROLE_CUDA(cudaMemcpy(volumes__d, volumes, sizeof(float) * PRIXS, cudaMemcpyHostToDevice));
	CONTROLE_CUDA(cudaMemcpy(   high__d,    high, sizeof(float) * PRIXS, cudaMemcpyHostToDevice));
	CONTROLE_CUDA(cudaMemcpy(    low__d,     low, sizeof(float) * PRIXS, cudaMemcpyHostToDevice));
	CONTROLE_CUDA(cudaMemcpy( median__d,  median, sizeof(float) * PRIXS, cudaMemcpyHostToDevice));
};

void     liberer_cudamalloc() {
	CONTROLE_CUDA(cudaFree(  prixs__d));
	CONTROLE_CUDA(cudaFree(volumes__d));
	CONTROLE_CUDA(cudaFree(   high__d));
	CONTROLE_CUDA(cudaFree(    low__d));
	CONTROLE_CUDA(cudaFree( median__d));
};

void charger_tout() {
	//	Assertions
	FOR(0, i, NATURES) ASSERT(nature_multiple_interv[i] <= MAX_MULTPLE_INTERV_NATURES);
	//
	printf("charger_les_prixs : ");    MESURER(charger_les_prixs());
	printf("charger_vram_nvidia : ");  MESURER(charger_vram_nvidia());
};

void liberer_tout() {
	titre("Liberer tout");
	liberer_cudamalloc();
};

ema_int_t * lire_ema_int(FILE * fp) {
	uint source, nature, K_ema, intervalle;
	uint params[MAX_PARAMS];
	FREAD(&source,     sizeof(uint), 1, fp);
	FREAD(&nature,     sizeof(uint), 1, fp);
	FREAD(&K_ema,      sizeof(uint), 1, fp);
	FREAD(&intervalle, sizeof(uint), 1, fp);
	//
	FREAD(&params,     sizeof(uint), MAX_PARAMS, fp);
	//
	return cree_ligne(source, nature, K_ema, intervalle, params);
};

void      ecrire_ema_int(ema_int_t * ema_int, FILE * fp) {
	FWRITE(&ema_int->source,     sizeof(uint), 1, fp);
	FWRITE(&ema_int->nature,     sizeof(uint), 1, fp);
	FWRITE(&ema_int->K_ema,      sizeof(uint), 1, fp);
	FWRITE(&ema_int->intervalle, sizeof(uint), 1, fp);
	//
	FWRITE(&ema_int->params,     sizeof(uint), MAX_PARAMS, fp);
};

char * nom_type_de_norme[3] = {
	"NORME_CLASSIQUE",
	"NORME_THEORIQUE",
	"NORME_RELATIVE "
};