#pragma once

#include "cuda.cuh"
#include "macro.cuh"

//  Outils
float   rnd();
float signe(float x);

void titre(char * str);

char * scientifique(uint nb);

//	template<typename T>
//		#include "impl_tmpl/tmpl_etc.cu"