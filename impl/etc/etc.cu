#include "main.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

float rnd()
{
#define PROFONDEURE 100000
	return (float)(rand() % PROFONDEURE) / (float)PROFONDEURE;
}

float signe(float x)
{
	return (x>=0 ? 1:-1);
};

char * scientifique(uint nb) {
	float x = (float)nb;
	uint dim = 0;
	uint s[100];
	while (x >= 1.0) {
		s[dim] = (uint)(x - (float)((uint)(x/10.0)*10.0));
		x /= 10.0;
		dim++;
	}
	//
	uint apostrophes = ((dim-1) - (dim-1)%(3))/3;
	char str[100];
	//
	uint pos_dans = 0;
	FOR(0, i, dim) {
		if (i % 3 == 0 && i != 0) {
			str[pos_dans] = '\'';
			pos_dans++;
		};
		str[pos_dans] = s[i] + '0';
		pos_dans++;
	}
	//
	char * inverse = (char*)malloc(pos_dans+1);
	FOR(0, i, pos_dans) inverse[i] = str[pos_dans-1-i];
	//
	inverse[pos_dans] = '\0';
	//
	return inverse;
};

double secondes()
{
	struct timespec now;
	timespec_get(&now, TIME_UTC);
	return 1000.0*(((int64_t) now.tv_sec) * 1000 + ((int64_t) now.tv_nsec) / 1000000);
};

PAS_OPTIMISER()
void titre(char * str) {
	printf("\033[93m=========\033[0m %s \033[93m=========\033[0m\n", str);
};