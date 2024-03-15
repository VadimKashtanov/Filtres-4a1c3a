#pragma once

#include "etc.cuh"

/*
Deux options d'optimisation s'offrent a moi
	1) depart -> depart + T
	2) [randint(depart, fin) for _ in range(T)]
*/

//	Si MODE==t_CONTINUE alors    t = depart+i
//	Si MODE==t_PSEUDO_ALEA alors t = pseudo(grain, i)

extern uint MODE_t_MODE;
extern uint grain_t_MODE;

#define t_CONTINUE    0
#define t_PSEUDO_ALEA 1

#define t_MODE(MODE, _depart, _fin, i, graine) ( \
	MODE == t_CONTINUE ? (_depart+i) : \
		( _depart+PSEUDO_ALEA((grain+i)) % (_fin-_depart) ))

#define t_MODE_GENERALE(_depart, _DEPART, _FIN, i) ( \
	MODE_t_MODE == t_CONTINUE ? t_MODE(t_CONTINUE, _depart, NULL, i, NULL) : \
	t_MODE(t_PSEUDO_ALEA, _DEPART, _FIN, i, grain_t_MODE)
)

#define cuda_t_MODE_GENERALE(MODE, GRAINE, _depart, _DEPART, _FIN, i) ( \
	MODE == t_CONTINUE ? t_MODE(t_CONTINUE, _depart, NULL, i, NULL) : \
	t_MODE(t_PSEUDO_ALEA, _DEPART, _FIN, i, GRAINE)
)

uint depart_plus_t = t_MODE_GENERALE(depart, DEPART, FIN, t);
uint cuda_depart_plus_t = cuda_t_MODE_GENERALE(_t_MODE, GRAINE, depart, DEPART, FIN, t);