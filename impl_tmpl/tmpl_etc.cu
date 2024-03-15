/*	Ceci est un fichier a integrer de maniere brute avec tous les templates avec #include 	*/
/*				Pas besoin de tete pour ces templates, mais c'est possible					*/
/*							C'est juste du copier-coller									*/
/*							Ca sera comme des fonctions static 								*/

#include "etc.cuh"

static __device__ float cuda_signe(float x) {
	//return (x>=0 ? 1:-1);
	if (x >= 0) return 1;
	else return -1;
}

//  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

template <typename T>
static T * alloc(uint A)
{
	return (T*)malloc(sizeof(T) * A);
}

template <typename T>
static T * liste(uint A, ...) {
	T val;
	va_list vl;
	va_start(vl,A);
	T * ret = alloc<T>(A);
	for (uint i=1; i < A; i++) ret[i] = va_arg(vl, T);
	va_end(vl);
	return ret;
};

static char * join(uint A, ...) {
	va_list vl;
	va_start(vl,A);
	char * liste[A];
	FOR(0, i, A) {
		liste[i] = va_arg(vl, char*);
	}
	va_end(vl);
	//
	uint taille = 1;
	FOR(0, i, A) taille += strlen(liste[i]);
	char * s = (char*)malloc(taille);
	uint depart = 0;
	FOR(0, i, A) {
		memcpy(s + depart, liste[i], strlen(liste[i]));
		depart += strlen(liste[i]);
	}
	s[taille-1] = '\0';
	//
	puts(s);
	return s;
}

template <typename T>
static T * copier(T * a, uint A)
{
	T * r = alloc<T>(A);
	memcpy(r, a, sizeof(T) * A);
	return r;
}

static float * lst_rnd(uint A, float a, float b)
{
	float * ret = alloc<float>(A);
	FOR(0, i, A) ret[i] = a + rnd()*(b-a);
	return ret;
}

template <typename T>
static T * zero(uint A)
{
	return (T*)calloc(A, sizeof(T));
}

//  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

template <typename T>
static T*   lire(char * fichier,        uint A)
{
	FILE * fp = fopen(fichier, "rb");
	T * ret = alloc<T>(A);
	fread(ret, sizeof(T), A, fp);
	fclose(fp);
	return ret;
}

template <typename T>
static void ecrire(char * fichier, T * l, uint A)
{
	FILE * fp = fopen(fichier, "wb");
	fwrite(l, sizeof(T), A, fp);
	fclose(fp);
}

//  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

template <typename T> 
static void cudafree(T * l)
{
	CONTROLE_CUDA(cudaFree(l));
}

template <typename T> 
static T *     cudalloc(                uint A)
{
	T * ret;
	CONTROLE_CUDA(cudaMalloc((void**)&ret, sizeof(T)*A));
	return ret;
}

template <typename T> 
static T *     cudazero(                uint A)
{
	T * ret = cudalloc<T>(A);
	CONTROLE_CUDA(cudaMemset(ret, 0, sizeof(T)*A));
	return ret;
}

template <typename T>
static T * cpu_vers_gpu(T * lst,        uint A)
{
	T * ret__d = cudalloc<T>(A);
	CONTROLE_CUDA(cudaMemcpy(ret__d, lst, sizeof(T)*A, cudaMemcpyHostToDevice));
	return ret__d;
}

template <typename T>
static T * gpu_vers_cpu(T * lst__d,     uint A)
{
	T * ret = alloc<T>(A);
	CONTROLE_CUDA(cudaMemcpy(ret, lst__d, sizeof(T)*A, cudaMemcpyDeviceToHost));
	return ret;
}

static void   cudaplume(float * lst__d, uint A)
{
	float * r = gpu_vers_cpu<float>(lst__d, A);
	FOR(0, i, A) printf("(%i)%f, ", i, r[i]);
	printf("\n");
	free(r);
}

static void mat_plume(float * lst, uint X, uint Y) {
	printf("   ");
	FOR(0, x, X) printf("     %03i  ", x);
	printf("\n");
	FOR(0, y, Y) {
		printf("%03i| ", y);
		FOR(0, x, X) printf("%+f ", lst[y*X+x]);
		printf("\n");
	}
}

template <typename T>
static void    cudazero(T * lst__d,     uint A)
{
	CONTROLE_CUDA(cudaMemset(lst__d, 0, sizeof(T)*A));
}

PAS_OPTIMISER()
static void comparer_lst(float * l0, float * l1, uint A, float profondeure) {
	FOR(0, i, A)
	{
		if (fabs(l0[i]-l1[i]) < profondeure) {
			printf("%03i| \033[92m%+f  ~= %+f\033[0m\n", i, l0[i], l1[i]);
		} else {
			printf("%03i| \033[91m%+f =/= %+f\033[0m \033[93m(dist=\033[91m%+f\033[93m)\033[0m\n", i, l0[i], l1[i], fabs(l0[i]-l1[i]));
		}
	};
};

static void comparer_lst_2d(
	float * l0, float * l1,
	uint X,
	uint Y, char * ynom,
	float profondeure)
{
	FOR(0, y, Y)
	{
		printf(" ### %s = %i ###\n", ynom, y);
		FOR(0, x, X)
		{
			uint i = y*X+x;
			if (fabs(l0[i]-l1[i]) < profondeure) {
				printf("%03i|\033[1m%03i\033[0m| \033[92m%+f  ~= %+f\033[0m\n",
					i, x, l0[i], l1[i]);
			} else {
				printf("%03i|\033[1m%03i\033[0m| \033[91m%+f =/= %+f\033[0m \033[93m(dist=\033[91m%+f\033[93m)\033[0m\n",
					i, x, l0[i], l1[i], fabs(l0[i]-l1[i]));
			}
		}
	}
};

static uint egales_lst(float * l0, float * l1, uint A, float profondeure) {
	FOR(0, i, A) {
		if (fabs(l0[i]-l1[i]) > profondeure)
			return 0;
	}
	return 1;
}

//	=============

static float * de_a(float de, float a, uint A) {
	float * ret = alloc<float>(A);
	FOR(0, i, A) ret[i] = de + (a-de)/(A-1)*i;
	return ret;
};