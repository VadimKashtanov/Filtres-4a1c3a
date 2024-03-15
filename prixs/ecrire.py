#! /usr/bin/python3

ema = lambda l,K,ema=0: [ema:=(ema*(1-1/K) + e/K) for e in l]

from sys import argv

prixs         = argv[1]
sortie_prixs  = "prixs.bin"		#close
sortie_volume = "volumes.bin"	#vBTC*close - vUSDT
sortie_hight  = "high.bin"		#high
sortie_low    = "low.bin"		#low
sortie_median = "median.bin"	#(haut+bas)/2

import struct as st

#	======== Lecture .csv ======= 
with open(prixs, "r") as co:
	text = co.read().split('\n')
	del text[0]
	del text[0]
	del text[-1]
	lignes = [l.split(',') for l in text][::-1] # <-- Important le [::-1] (car les prixs sont du plus recent au plus ancien)
	infos = [(float(Close), float(Volume_BTC), float(Volume_USDT), float(Low), float(High)) for Unix,Date,Symbol,Open,High,Low,Close,Volume_BTC,Volume_USDT,tradecount in lignes]

#	========= Ecriture ==========
prixs   = [p                  for  p,__,__,__,__ in infos]
_low    = [l                  for __,__,__, l,__ in infos]
_hight  = [h                  for __,__,__,__, h in infos]
s=0; volumes = [s:=(s + (vb*p-vu)) for  p,vb,vu,__,__ in infos]
median  = [(h+l)/2            for __,__,__, l, h in infos]

with open(sortie_prixs, "wb") as co:
	print(f"LEN prixs   = {len(prixs)}")
	co.write(st.pack('I', len(prixs)))
	co.write(st.pack('f'*len(prixs), *prixs))

with open(sortie_volume, "wb") as co:
	print(f"LEN volumes = {len(volumes)}")
	co.write(st.pack('I', len(volumes)))
	co.write(st.pack('f'*len(volumes), *volumes))
	
with open(sortie_hight, "wb") as co:
	print(f"LEN hight   = {len(_hight)}")
	co.write(st.pack('I', len(_hight)))
	co.write(st.pack('f'*len(_hight), *_hight))

with open(sortie_low,   "wb") as co:
	print(f"LEN low     = {len(_low)}")
	co.write(st.pack('I', len(_low)))
	co.write(st.pack('f'*len(_low), *_low))

with open(sortie_median,   "wb") as co:
	print(f"LEN median  = {len(median)}")
	co.write(st.pack('I', len(median)))
	co.write(st.pack('f'*len(median), *median))