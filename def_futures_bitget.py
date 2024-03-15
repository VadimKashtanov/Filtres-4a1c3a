from pybitget import Client
from os import system
from time import time
import importlib
import struct as st
from time import sleep
from datetime import datetime
from os import listdir
from importlib import reload

SYMBOLE     = 'BTCUSDT_UMCBL'
productType = 'umcbl'
marginCoin  = 'USDT'

def fermer_les_positions():
	try:
		params = {'productType':productType}
		client._request_with_params("POST",  '/api/mix/v1/order' + '/close-all-positions', params)
	except Exception as E:
		print(E)

def executer_future(client, pourcent, signe):
	usdt = str(argent*pourcent)
	print(f"{('Achat' if signe==+1 else 'Vente')} de {usdt}")

	SIZE = (pourcent if signe==+1 else argent/float(client.mix_get_market_price('BTCUSDT_UMCBL')['data']['markPrice'])/pourcent)
	print(SIZE)
	data = client.mix_place_order(
		symbol=SYMBOLE,
		marginCoin=marginCoin,
		size=SIZE,
		side=('open_long' if signe==+1 else 'open_short'),
		orderType='market')
	print(data)

def faire_prediction():
	system("python3 __bitget__mdl_prediction.py resultat_mdl_pred")
	with open("resultat_mdl_pred", 'r') as co:
		return eval(co.read())

def derniere_prediction():
	if not 'predictions_mdl' in listdir():
		with open('predictions_mdl', 'w') as co:
			la = datetime.now()
			co.write(str({(la.hour, la.day, la.month) : faire_prediction()}))

	with open('predictions_mdl', 'r') as co:
		dico = eval(co.read())
	la = datetime.now()
	if (la.hour, la.day, la.month) in list(dico.keys()):
		return dico[(la.hour, la.day, la.month)]
	else:
		with open('predictions_mdl', 'w') as co:
			co.write(
				str(
					{**dico, (la.hour, la.day, la.month) : faire_prediction()}
				)
			)