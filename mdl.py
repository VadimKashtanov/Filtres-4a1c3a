import struct as st

from donnees_bitget import *

from os import system

import numba
from numba import njit, prange

from math import exp, tanh
import struct as st

from random import random, randint

rnd = lambda : 2*random()-1

def lire_uint(I, _bin):
	l = list(st.unpack('I'*I, _bin[:st.calcsize('I')*I]))
	return l, _bin[st.calcsize('I')*I:]

def lire_flotants(I, _bin):
	l = list(st.unpack('f'*I, _bin[:st.calcsize('f')*I]))
	return l, _bin[st.calcsize('f')*I:]

def norme(arr):
	_min = min(arr)
	_max = max(arr)
	return [(e-_min)/(_max-_min) for e in arr]

def e_norme(arr):
	_min = min(arr)
	_max = max(arr)
	return [2*(e-_min)/(_max-_min)-1 for e in arr]

def ema(arr, K):
	e = [arr[0]]
	for p in arr[1:]:
		e += [e[-1]*(1-1/(1+K)) + p*1/(1+K)]
	return e

with open("structure_generale.bin", "rb") as co:
	bins = co.read()

	constantes, bins = lire_uint(16, bins)

	exec("""DEPART,
		P,
		P_INTERV,
		N,
		MAX_INTERVALLES,
		SOURCES,
		MAX_PARAMS, NATURES,
		MAX_EMA, MAX_PLUS, MAX_COEF_MACD,
		C, MAX_Y, BLOQUES, F_PAR_BLOQUES,
		INSTS
		""".replace('\n', '') + " = constantes")

	min_param, bins = lire_uint(NATURES*MAX_PARAMS, bins)
	max_param, bins = lire_uint(NATURES*MAX_PARAMS, bins)

	NATURE_PARAMS, bins = lire_uint(NATURES, bins)

	#assert DEPART >= __DEPART

MIN_EMA = 1
MIN_NATURES = 0
MIN_NATURES = NATURES-1
MIN_INTERVALLES = 1

#DEPART = N*MAX_INTERVALLES

########################

def filtre_prixs__poids(Y,X):
	return F_PAR_BLOQUES*BLOQUES*N#Y*N

def filtre_prixs_tolerant__poids(Y,X):
	return F_PAR_BLOQUES*BLOQUES*N#Y*N

def dot1d__poids(Y,X):
	return (X+1)*Y

########################

inst_poids = [
	filtre_prixs__poids,
	filtre_prixs_tolerant__poids,
	dot1d__poids,
	0,
	0
]

########################

NORME_CLASSIQUE = 0 #return brute, None,            type_norme
NORME_THEORIQUE = 1 #return brute, borne_theorique, type_norme
NORME_RELATIVE  = 2 #return brute, None,            type_norme

### Natures ###
#@njit(parallel=True)
def __nature__ema(source, params):
	return source, None, NORME_CLASSIQUE

#@njit(parallel=True)
def __nature__macd(source, params):
	coef,_,_,_ = params
	#
	assert coef > 0.0
	ema12 = ema(source, 12*coef);
	ema26 = ema(source, 26*coef);
	_macd = [ema12[i]-ema26[i] for i in range(len(source))]
	ema9  = ema(_macd,  12*coef);
	return [_macd[i] - ema9[i] for i in range(len(source))], None, NORME_RELATIVE

#@njit(parallel=True)
def __nature__chiffre(source, params):
	D,_,_,_ = params
	return [2*(D-min([abs(x-D*round((x+0)/D)), abs(x-D*round((x+D)/D))]))/D-1 for x in source], (0, D/2), NORME_THEORIQUE

#@njit(parallel=True)
def __nature__awesome(source, params):
	coef,_,_,_ = params
	#
	assert coef > 0.0
	ema5  = ema(source,  5*coef)
	ema34 = ema(source, 34*coef)
	return [ema5[i] - ema34[i] for i in range(len(source))], None, NORME_RELATIVE

def __nature__pourcent(source, params):
	interv,K_post_pr,_,_ = params
	#
	n = 14*interv
	#
	pourcent_r = [0 for _ in source]
	#
	for i in range(n, len(source)):
		_max = source[i]
		_min = source[i]
		for j in range(n):
			if (source[i-j] > _max): _max = source[i-j];
			if (source[i-j] < _min): _min = source[i-j];
		pourcent_r[i] = (_max - source[i])/(_max - _min)  * (-1);
	#
	return ema(pourcent_r, K_post_pr), (-1,0), NORME_THEORIQUE

def __nature__rsi(source, params):
	interv,_,_,_ = params
	#
	n = 14 * interv
	#
	rsi = [0 for _ in source]
	changements  = [0] + [a-b for a,b in zip(source[1:], source)] 
	#
	for t in range(n+1, len(source)):
		gain_moy = 0;
		perte_moy = 0;
		for i in range(n):
			if (changements[t-i] >= 0): gain_moy  += +(changements[t-i]);
			if (changements[t-i] <  0): perte_moy += -(changements[t-i]);
		if perte_moy > 0:
			rs = (gain_moy/n) / (perte_moy/n)
			rsi[t] = 1.0 - 1.0 / (1 + rs)
		else:
			rsi[t] = 1.0;
	#
	return rsi, (0,1), NORME_THEORIQUE

I__natures = [
	__nature__ema,
	__nature__macd,
	__nature__chiffre,
	__nature__awesome,
	__nature__pourcent,
	__nature__rsi
]

########################

class Mdl:
	def __init__(self, fichier):
		with open(fichier, "rb") as co:
			bins = co.read()

		self.Y,             bins = lire_uint(C, bins)
		self.insts,         bins = lire_uint(C, bins)
		self.cibles,        bins = lire_uint(C, bins)
		self.decale_future, bins = lire_uint(C, bins)

		for inst in self.insts:
			assert inst <= 3

		self.ema_int = []
		self.lignes  = []
		for blk in range(BLOQUES):
			(source, nature, K_ema, intervalle), bins = lire_uint(4, bins)
			params, bins = lire_uint(MAX_PARAMS, bins)

			self.ema_int += [{
				'source'     : source,
				'nature'     : nature,
				'K_ema'      : K_ema,
				'intervalle' : intervalle,
				'params'     : params
			}]
			
			assert nature <= max([0,1,2,3,4,5])

			self.lignes += [
				I__natures[nature]( ema(I__sources[source], K_ema), params)
			]

			#if (blk == 900):
			#	plt.plot(self.lignes[-1][0]); plt.show()

		#breakpoint()
		#i=480;print(self.ema_int[i]);K=self.ema_int[i]['intervalle'];plt.plot(self.lignes[i][0][-1000:]);plt.plot([0]*K*8);plt.show()
		
		self.p = []
		self.poids = []
		for i in range(C):
			X, Y = (self.Y[i-1] if i!=0 else 0), self.Y[i]
			self.poids += [inst_poids[self.insts[i]](Y,X)]
			poids, bins = lire_flotants(self.poids[i], bins)
			self.p += [poids]

	def incruster_DOT1D(self, apres, Y):
		global C
		print(f"Désormais C={len(self.Y)+1} au lieux de C={C}")
		C += 1
		#
		X = self.Y[apres]
		p = [(2*rnd()-1) * 0.30 for _ in range((X+1)*Y)]
		#
		cibles = 0
		decale_future = 0
		#
		self.insts = self.insts[:apres] + [   1  ] + self.insts[apres:]
		self.Y     = self.Y    [:apres] + [   Y  ] + self.Y    [apres:]
		#
		self.cibles        = self.cibles       [:apres] + [   cibles         ] + self.cibles       [apres:]
		self.decale_future = self.decale_future[:apres] + [   decale_future  ] + self.decale_future[apres:]
		#
		self.poids = self.poids[:apres] + [len(p)] + self.poids[apres:]
		self.p     = self.p    [:apres] + [   p  ] + self.p    [apres:]

	def ecrire(self, fichier):
		with open(fichier, "wb") as co:
			co.write(st.pack('I'*C, *self.Y))
			co.write(st.pack('I'*C, *self.insts))
			co.write(st.pack('I'*C, *self.cibles))
			co.write(st.pack('I'*C, *self.decale_future))
			#
			for ema_int in self.ema_int:
				co.write(st.pack(
					'I'*4,
					ema_int['source'],
					ema_int['nature'],
					ema_int['K_ema'],
					ema_int['intervalle']
				))
				co.write(st.pack('I'*MAX_PARAMS, *ema_int['params']))
			#
			for i in range(len(self.p)):
				co.write(st.pack('f'*self.poids[i], *self.p[i]))

	def __call__(self):
		assert len(I__sources[0]) != 0
		assert len(I__sources[0]) == len(I__sources[1]) == len(I__sources[2]) == len(I__sources[3])

		with open("communication.bin", "wb") as co:
			co.write(st.pack('I'*len(self.Y), *self.Y))
			co.write(st.pack('I'*len(self.insts), *self.insts))
			co.write(st.pack('I'*len(self.Y), *self.cibles))
			co.write(st.pack('I'*len(self.insts), *self.decale_future))
			#
			co.write(st.pack('I', I_PRIXS))
			#
			co.write(st.pack('I'*BLOQUES, *list(map(lambda x:x['intervalle'], self.ema_int))))
			#
			type_norme = []
			min_norme, max_norme = [], []
			for _,params,type_de_norme in self.lignes:
				if type_de_norme == NORME_CLASSIQUE:
					type_norme += [NORME_CLASSIQUE]
					min_norme  += [0]
					max_norme  += [0]
				elif type_de_norme == NORME_THEORIQUE:
					type_norme += [NORME_THEORIQUE]
					min_norme  += [params[0]]
					max_norme  += [params[1]]
				elif type_de_norme == NORME_RELATIVE:
					type_norme += [NORME_RELATIVE]
					min_norme  += [0]
					max_norme  += [0]
				else:
					raise Exception(f"Pas de norme {type_de_norme}")
			co.write(st.pack('I'*BLOQUES, *type_norme))
			co.write(st.pack('f'*BLOQUES, *min_norme ))
			co.write(st.pack('f'*BLOQUES, *max_norme ))
			#
			for ligne,_,_ in self.lignes:
				co.write(st.pack('f'*I_PRIXS, *ligne))
			#
			for poids in self.p:
				co.write(st.pack('I', len(poids)))
				co.write(st.pack('f'*len(poids), *poids))
		#
		#system("gdb --args ./prog4__simple_mdl_pour_python communication.bin")
		system("./prog4__simple_mdl_pour_python communication.bin")
		#
		with open("communication.bin", "rb") as co:
			bins = co.read()
			ret, bins = lire_flotants(I_PRIXS-DEPART, bins)

		return ret