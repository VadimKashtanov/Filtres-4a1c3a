from pybitget import Client
from os import system
from time import time
import importlib
import struct as st
from time import sleep
from datetime import datetime
from os import listdir

from def_futures_bitget import fermer_les_positions, executer_future, derniere_prediction

client = Client(
	'bg_b13daa6f909560dd8dfac7549e9ce83f',
	'd1c9e1c5c520a74b89133945f79246929e27908187cdb2fd51bab280b0dd98e9',
	'Henrya10ans')

argent = lambda : float(client.mix_get_account('BTCUSDT_UMCBL', 'USDT')['data']['usdtEquity'])

print(f"Le compte a {argent()} $")

SYMBOLE     = 'BTCUSDT_UMCBL'
productType = 'umcbl'
marginCoin  = 'USDT'

fermer_les_positions()

signe = lambda x: (1 if x >= 0 else -1)

#executer_future(client, POURCENT*abs(pred[-1]), signe(pred[-1]))

Mois = 'Janvier', 'Fevrier', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet', 'Aout', 'Septembre', 'Octobre', 'Novembre', 'Decembre'

from mdl import *

__pred = derniere_prediction()
#executer_future(client, POURCENT*abs(__pred), signe(__pred))

'''
while True:
	la = datetime.now()
	print(f"[{la.hour}h {la.day} {Mois[la.month-1]}] {argent()}$")
	for i in range(60): # 60 minutes
		ordres = client.mix_get_all_open_orders(productType)['data']['orderList']
		assert len(ordres) in (0,1)
		#
		if len(ordres) == 1:
			ordre = ordres[0]
			prix = float(client.mix_get_order_details(
				SYMBOLE,
				orderId=ordre['orderId'],
				clientOrderId=ordre['clientOid']
			)['data']['price'])

			prix_actuel = float(client.mix_get_market_price('BTCUSDT_UMCBL')['data']['markPrice'])

			derniere_pred = dernier_prediction()

			if ((prix_actual/prix-1)*LEVIER*POURCENT*derniere_pred < -1):
				fermer_les_positions()
			
		#
		sleep(1)#sleep(60) #Attendre 1 minute

	#   Poser la mise en $
	executer_future(client, POURCENT*abs(pred[-1]), signe(pred[-1]))

'''







