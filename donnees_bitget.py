import time
import datetime

import requests

import matplotlib.pyplot as plt

ARONDIRE_AU_MODULO = lambda x,mod: (x + (mod - (x%mod)) if x%mod!=0 else x)

milliseconde = lambda la: int(la * 1000   )*1
seconde      = lambda la: int(la          )*1000
heure        = lambda la: int(la / (60*60))*1000*60*60

requette_bitget = lambda de, a: eval(
	requests.get(
		f"https://api.bitget.com/api/mix/v1/market/history-candles?symbol=BTCUSDT_UMCBL&granularity=1H&startTime={de}&endTime={a}"
	).text
)

HEURES_PAR_REQUETTE = 100

T = (4)*7*24

__DEPART = (15+0)*256

la = heure(time.time())
heures_voulues = [
	la - 60*60*1000*i
	for i in range(ARONDIRE_AU_MODULO(__DEPART+T, HEURES_PAR_REQUETTE))
][::-1] # <<<< ---- ??????

donnees = []

REQUETTES = int(len(heures_voulues) / HEURES_PAR_REQUETTE)
print(f"Extraction de {len(heures_voulues)} heures depuis api.bitget.com ...")
for i in range(REQUETTES):
	paquet_heures = requette_bitget(heures_voulues[i*HEURES_PAR_REQUETTE], heures_voulues[(i+1)*HEURES_PAR_REQUETTE-1])
	donnees += paquet_heures

	if i % 1 == 0:
		print(f"[{round(i*HEURES_PAR_REQUETTE/len(heures_voulues)*100)}%],   len(paquet_heures)={len(paquet_heures)}")

#donnees = donnees[::-1]

print(f"HEURES VOULUES = {len(heures_voulues)}, len(donnees)={len(donnees)}")

prixs   = [float(c)                       for _,o,h,l,c,vB,vU in donnees]
hight   = [float(h)                       for _,o,h,l,c,vB,vU in donnees]
low     = [float(l)                       for _,o,h,l,c,vB,vU in donnees]
volumes = [float(c)*float(vB) - float(vU) for _,o,h,l,c,vB,vU in donnees]
median  = [(float(h)+float(l))/2          for _,o,h,l,c,vB,vU in donnees]

I_PRIXS = len(prixs)

I__sources = [
	prixs,
	volumes,
	hight,
	low,
	median
]

print("5 dernieres heures : ", prixs[-5:])

#plt.plot(prixs);plt.show()