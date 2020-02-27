# Autor: Alexandre Luis Debiasi Gandini
# jan/2020
# codigo de suporte ao artigo da disciplina de series temporais - mestrado em estatistica

%matplotlib
import os
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import datetime
import pickle
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.vector_ar.vecm import select_order
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.vecm import select_coint_rank
import itertools
from statsmodels.tsa.stattools import coint
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller

# funcoes preliminares:

def grava(coisa,filename,path): #formato do path:'/Users/Alexandre Gandini/OneDrive/Data Science/Calcular percent move ATR channel/'
	pkl_file = open(path + filename, 'wb')
	pickle.dump(coisa, pkl_file)
	pkl_file.close()

def abre(filename,path): #formato do path:'/Users/Alexandre Gandini/OneDrive/Data Science/Calcular percent move ATR channel/'
	pkl_file = open(path + filename, 'rb')
	coisa = pickle.load(pkl_file)
	pkl_file.close()
	return coisa

def descobre_arquivos_na_pasta(pasta,tipo_do_arquivo='.xlsx'):
	#Descobre arquivos na pasta:
	arquivos = []
	for file in os.listdir(pasta):
		arquivos.append(os.fsdecode(file))
	arquivos = [arquivo for arquivo in arquivos if tipo_do_arquivo in arquivo] #seleciona soh arquivos com .xlsx
	return arquivos

def date_range(start, end, intv):
    from datetime import datetime
    start = datetime.strptime(start,"%Y%m%d")
    end = datetime.strptime(end,"%Y%m%d")
    diff = (end  - start ) / intv
    for i in range(intv):
        yield (start + diff * i).strftime("%Y%m%d")
    yield end.strftime("%Y%m%d")

def moving_date_range(start, end, jump=22):
    from datetime import datetime, timedelta
    from datetime import timedelta
	start = datetime.strptime(start,"%Y%m%d")
    end = datetime.strptime(end,"%Y%m%d")
    day = start
	jump = timedelta(jump)
	
	seq = []
	seq.append(start.strftime("%Y%m%d"))

	while day < end:
		day = day + jump
		seq.append(day.strftime("%Y%m%d"))
	return seq


def normalize_and_plot(close,tickers,log=True,start_level=1):
	if log:
		return ((np.log(close[tickers]).diff().fillna(0) + 1).cumprod() + (start_level - 1)).plot()
	else:
		return (((close[tickers]).pct_change().fillna(0) + 1).cumprod() + (start_level - 1)).plot()

# a cada periodo recomeca em 1:
def normalize_reestart_and_plot(close,tickers,datas_limite,log=True,start_level=1):
	result_df = DataFrame(columns=tickers)
	for date in datas_limite:
		if date != datas_limite[-1]: # faz ateh a penultima:
			if log:
				result_df = result_df.append((np.log(close[tickers][date:datas_limite[datas_limite.index(date)+1]]).diff().fillna(0) + 1).cumprod() + (start_level - 1))
			else:
				result_df = result_df.append(((close[tickers][date:datas_limite[datas_limite.index(date)+1]]).pct_change().fillna(0) + 1).cumprod() + (start_level - 1))
			result_df['start level'] = start_level
	return result_df.plot()

# leitura dos dados:

# cria base de dados a partir de csvs da quotebr:
# cada entrada do dicionario base eh uma acao.

#agora dois a dois, medindo a "forca" da cointegracao pelo qual negativo eh o teste ADF de unit root nos residuos:

data_folder = 'C:/Users/alega/Documents/Mestrado_stats/series_temp/artigo/data/'
arquivos = descobre_arquivos_na_pasta(data_folder,tipo_do_arquivo='.csv')
base = {}
arquivos = descobre_arquivos_na_pasta(data_folder,tipo_do_arquivo='.csv')
for arquivo in tqdm(arquivos):
    base[arquivo[:-4]] = pd.read_csv(data_folder + arquivo, parse_dates=['Date'],index_col='Date')
    base[arquivo[:-4]]['Adj Close'] = base[arquivo[:-4]]['Close']
    base[arquivo[:-4]] = base[arquivo[:-4]][['High','Open','Low','Close','Volume','Adj Close','Open Interest']]

#reduz tamanho da base, comeca em determinado ano:
# ano_inicio = '2018-01'
# ano_inicio = '2016-01'
ano_inicio = '20050101'
# ano_inicio = '20150101'
ano_fim = '20191231'
for ticker in base:
    base[ticker] = base[ticker][ano_inicio:ano_fim]

qtd_date_chunks = 10
datas_limite = list(date_range(ano_inicio, ano_fim, qtd_date_chunks))

#deleta $PTAX:
if '$PTAX' in base:
    del base['$PTAX']

if 'CPFE3' in base:
	del base['CPFE3'] # descontinuidade na serie

#retira os tickers com dois numeros no final, mantem os 11:
tickers_excluir = []
for ticker in base:
    # if ticker[-2:] == '11':
    #     pass
    # agora tiro TODOS, inclusive os final 11 (tem muito lixo ali) - agora tira os final 11 tb:
    if (ticker[-2:].isdigit()) or (ticker[-3:] == '11B') :
        tickers_excluir.append(ticker)

for ticker in tickers_excluir:
    del base[ticker]

pd.options.mode.chained_assignment = None  # default='warn'

ticker_mais_antigo = 'PETR4'

# dfs close:
close = DataFrame(0,columns=base.keys(),index=base[ticker_mais_antigo].index)
for ticker in tqdm(base):
    close[ticker] = base[ticker]['Close']

# retira todos os ativos que nao tiveram negocio (nan) em algum dia dos dez anos - tanto os que comecaram depois quanto os que foram extintos antes.
close = close.dropna(axis=1)

sign_level = 0.05
# sign_level = 0.01
# sign_level = 0.1

k = 2
# combinations = list(itertools.combinations(close.columns[:30], k))
combinations = list(itertools.combinations(close.columns[:], k))

print('quantidade de combinacoes de',str(k),'tickers:',len(combinations))

# Primeiro testa pra todo o periodo:

results = []

for comb in tqdm(combinations):
	new_close = np.log(close[[item for item in comb]])[ano_inicio:ano_fim]
	# new_close = (close[[item for item in comb]])

	cointegra = coint(new_close.iloc[:,0],new_close.iloc[:,1])
	t_stat = cointegra[0]
	p_value = cointegra[1]
	
	if p_value <= sign_level: # cointegram:

		results.append((t_stat, p_value, comb[0], comb[1], 'all' ))

results_df = DataFrame(results,columns=['adf','p_value','ticker0','ticker1','date']).sort_values(by=['adf'], ascending=[True])
print(results_df.head(50))

#random plot:
sample = results_df.sample()[['ticker0','ticker1']].iloc[0].to_list()
# escolher tickers plot:
# ticker0 = 'DTEX3'
# ticker1 = 'ODPV3'
# sample = [ticker0, ticker1]
ax = normalize_and_plot(close,list(sample)) # plot normal
ax2 = normalize_reestart_and_plot(close,list(sample),datas_limite) # plot recomecando em start level (1) a cada data limite
ticker0 = sample[0]
ticker1 = sample[1]
# gray areas per periods:
for i in range(len(datas_limite)-1):
	if results_df[(results_df['ticker0'] == ticker0) & (results_df['ticker1'] == ticker1)]['date'].str.contains(datas_limite[i]).sum() > 0:
		color = 'lightgray'
	else:
		color = 'gray'
	ax.axvspan(datas_limite[i], datas_limite[i+1] , alpha=0.5, color=color)
	ax2.axvspan(datas_limite[i], datas_limite[i+1] , alpha=0.5, color=color)


# agora soh as combinacoes que passaram no teste da cointegracao do periodo todo

combinations = [(ticker0, ticker1) for ticker0, ticker1 in zip(results_df['ticker0'], results_df['ticker1'])]

results = []

for date in tqdm(datas_limite):

	if date == datas_limite[-1]: # se for a ultima:
		break
	else:
		for comb in tqdm(combinations):
			new_close = np.log(close[[item for item in comb]])[date:datas_limite[datas_limite.index(date) + 1]]
			# new_close = (close[[item for item in comb]])

			cointegra = coint(new_close.iloc[:,0],new_close.iloc[:,1])
			t_stat = cointegra[0]
			p_value = cointegra[1]
			
			if p_value <= sign_level: # cointegram:

				results.append((t_stat, p_value, comb[0], comb[1], date ))

results_df = DataFrame(results,columns=['adf','p_value','ticker0','ticker1','date']).sort_values(by=['adf'], ascending=[True])
print(results_df.head(50))
agrupados_count = results_df.groupby(['ticker0','ticker1']).count().sort_values('adf',ascending=False)
print(agrupados_count)

most_windows = agrupados_count.iloc[0]['adf']
melhores = agrupados_count[agrupados_count['adf'] == most_windows].index
agrupados_mean = results_df.groupby(['ticker0','ticker1']).mean().sort_values('adf',ascending=True)
melhores = agrupados_mean.loc[melhores].sort_values('adf',ascending=True)
melhores['qtd_windows'] = agrupados_count['adf']
print(melhores)


#random plot:
sample = melhores.sample().index[0]
# escolher tickers plot:
# ticker0 = 'DTEX3'
# ticker1 = 'ODPV3'
# sample = [ticker0, ticker1]
ax = normalize_and_plot(close,list(sample)) # plot normal
ax2 = normalize_reestart_and_plot(close,list(sample),datas_limite) # plot recomecando em start level (1) a cada data limite
ticker0 = sample[0]
ticker1 = sample[1]
# gray areas per periods:
for i in range(len(datas_limite)-1):
	if results_df[(results_df['ticker0'] == ticker0) & (results_df['ticker1'] == ticker1)]['date'].str.contains(datas_limite[i]).sum() > 0:
		color = 'lightgray'
	else:
		color = 'gray'
	ax.axvspan(datas_limite[i], datas_limite[i+1] , alpha=0.5, color=color)
	ax2.axvspan(datas_limite[i], datas_limite[i+1] , alpha=0.5, color=color)





##############

# agora com janelas moveis, pra ver quem tem mais janelas de cointegracao.

datas_limite = moving_date_range(ano_inicio, ano_fim, jump=22)

results = []

datas_limite_a_frente = 36

datas_limite_a_pular = 12

cont = 0

for date in tqdm(datas_limite):

	if date == datas_limite[-datas_limite_a_frente]: # se for a ultima:
		break
	
	elif (cont % datas_limite_a_pular) != 0: # pulo da janela
		pass

	else:
		for comb in tqdm(combinations):
			new_close = np.log(close[[item for item in comb]])[date:datas_limite[datas_limite.index(date) + datas_limite_a_frente]]
			# new_close = (close[[item for item in comb]])

			cointegra = coint(new_close.iloc[:,0],new_close.iloc[:,1])
			t_stat = cointegra[0]
			p_value = cointegra[1]
			
			if p_value <= sign_level: # cointegram:

				results.append((t_stat, p_value, comb[0], comb[1], date ))
	
	cont += 1

results_df = DataFrame(results,columns=['adf','p_value','ticker0','ticker1','date']).sort_values(by=['adf'], ascending=[True])
print(results_df.head(50))
agrupados_count = results_df.groupby(['ticker0','ticker1']).count().sort_values('adf',ascending=False)
print(agrupados_count)

most_windows = agrupados_count.iloc[0]['adf']
melhores = agrupados_count[agrupados_count['adf'] == most_windows].index
agrupados_mean = results_df.groupby(['ticker0','ticker1']).mean().sort_values('adf',ascending=True)
melhores = agrupados_mean.loc[melhores].sort_values('adf',ascending=True)
melhores['qtd_windows'] = agrupados_count['adf']
print(melhores)


#random plot:
sample = melhores.sample().index[0]
# escolher tickers plot:
ticker0 = 'CCRO3'
ticker1 = 'ITSA4'
sample = [ticker0, ticker1]
ax = normalize_and_plot(close,list(sample)) # plot normal
# ax2 = normalize_reestart_and_plot(close,list(sample),datas_limite) # plot recomecando em start level (1) a cada data limite
ticker0 = sample[0]
ticker1 = sample[1]
# gray areas per periods:
for i in range(len(datas_limite)-datas_limite_a_frente):
	if results_df[(results_df['ticker0'] == ticker0) & (results_df['ticker1'] == ticker1)]['date'].str.contains(datas_limite[i]).sum() > 0:
		color = 'lightgray'
		ax.axvspan(datas_limite[i], datas_limite[i+datas_limite_a_frente] , alpha=0.5, color=color)
		# ax2.axvspan(datas_limite[i], datas_limite[i+datas_limite_a_frente] , alpha=0.5, color=color)
	# else:
	# 	color = 'gray'


### escolhidos pro artigo:
ticker0 = 'ABEV3'
ticker1 = 'UGPA3'

ticker0 = 'BBAS3'
ticker1 = 'BRAP4'

ticker0 = 'CCRO3'
ticker1 = 'ITSA4'
















agrupados_count = results_df.groupby(['ticker0','ticker1']).count().sort_values('adf',ascending=False)
print(agrupados_count)

most_windows = agrupados_count.iloc[0]['adf']
melhores = agrupados_count[agrupados_count['adf'] == most_windows].index
agrupados_mean = results_df.groupby(['ticker0','ticker1']).mean().sort_values('adf',ascending=True)
melhores = agrupados_mean.loc[melhores].sort_values('adf',ascending=True)
melhores['qtd_windows'] = agrupados_count['adf']
print(melhores)



















############ antigo:

for comb in tqdm(combinations):
	new_close = np.log(close[[item for item in comb]])[ano_inicio:ano_fim]
	# new_close = (close[[item for item in comb]])

	p_value = (coint(new_close.iloc[:,0],new_close.iloc[:,1]))[1]

	ols = smf.ols(new_close.columns[0] + ' ~ ' + new_close.columns[1],data=new_close).fit()
	resid = ols.resid
	adf = adfuller(resid,maxlag=1)
	
	if adf[1] <= sign_level: #there is a unit root:

		results.append((adf[0],adf[1], adf[2], [item for item in comb], 'all' ))
		# print((select_rank.rank, [item for item in comb]))

results_df2 = DataFrame(results,columns=['adf','p_value','usedlag','tickers','date']).sort_values(by=['adf','usedlag'], ascending=[True, True])
print(results_df2.head(50))


#################

sign_level = 0.05
sign_level = 0.1

k = 2
combinations = list(itertools.combinations(close.columns[100:200], k))
# combinations = list(itertools.combinations(close.columns[:], k))

print('quantidade de combinacoes de',str(k),'tickers:',len(combinations))

results = []

for comb in tqdm(combinations):
	new_close = np.log(close[[item for item in comb]])
	# new_close = (close[[item for item in comb]])

	p_value = (coint(new_close.iloc[:,0],new_close.iloc[:,1]))[1]

	ols = smf.ols(new_close.columns[0] + ' ~ ' + new_close.columns[1],data=new_close).fit()
	resid = ols.resid
	adf = adfuller(resid,maxlag=1)
	
	if adf[1] <= sign_level: #there is a unit root:

		results.append((adf[0],adf[1], adf[2], [item for item in comb] ))
		# print((select_rank.rank, [item for item in comb]))

print(results)
print(len(results))
results_df = DataFrame(results,columns=['adf','p_value','usedlag','tickers']).sort_values(by=['adf','usedlag'], ascending=[True, True])
print(results_df.head(50))


def normalize_and_plot(close,tickers,log=True,start_level=1):
	if log:
		return ((np.log(close[tickers]).diff().fillna(0) + 1).cumprod() + (start_level - 1)).plot()
	else:
		return (((close[tickers]).pct_change().fillna(0) + 1).cumprod() + (start_level - 1)).plot()

normalize_and_plot(close,results_df['tickers'].iloc[0])

#random plot:
normalize_and_plot(close,results_df.sample()['tickers'].iloc[0])




normalize_and_plot(close,['ABEV3','ALSO3','BBSE3'])
normalize_and_plot(close,['ABCB4', 'ANIM3', 'APER3'])
normalize_and_plot(close,['ABCB4', 'AZEV4', 'APER3'])


normalize_and_plot(close,['AGRO3','CESP6','CPLE3'],log=False)
normalize_and_plot(close,['ABEV3','ALSO3','BBSE3'])
normalize_and_plot(close,['ABCB4','ALSO3','BBSE3'])


normalize_and_plot(close,['ALPA4', 'BRSR6', 'CARD3'],log=False)
normalize_and_plot(close,['ALPA4', 'BRSR6', 'CARD3'])
normalize_and_plot(close,['BBAS3', 'BRKM5', 'CPLE6'])

normalize_and_plot(close,['STBP3', 'TRIS3'],log=False)
normalize_and_plot(close,['STBP3', 'TRIS3'])
normalize_and_plot(close,['LUPA3', 'USIM5'])
normalize_and_plot(close,['POMO3', 'RAPT4'])
normalize_and_plot(close,['PETR3', 'TGMA3'])
normalize_and_plot(close,['RAIL3','RLOG3'])
normalize_and_plot(close,['LUPA3', 'MGLU3', 'MILS3', 'MOVI3', 'MULT3', 'ODPV3'])
normalize_and_plot(close,['LUPA3','MDIA3'])
normalize_and_plot(close,['PDGR3', 'PETR3', 'PETR4'])
normalize_and_plot(close,['MEAL3', 'MGLU3', 'MILS3', 'MULT3', 'ODPV3', 'OIBR3'])
normalize_and_plot(close,['MEAL3', 'MILS3', 'MULT3', 'ODPV3'])
normalize_and_plot(close,['LUPA3', 'MDIA3', 'MGLU3', 'MILS3', 'MOVI3', 'OFSA3'])
normalize_and_plot(close,['MILS3', 'MOVI3', 'POMO4'])
normalize_and_plot(close,['LUPA3', 'MDIA3', 'MGLU3', 'MILS3', 'MRFG3', 'ODPV3'])
normalize_and_plot(close,['ARZZ3', 'BRAP4', 'CAML3'])
normalize_and_plot(close,['B3SA3', 'BRAP4', 'CPLE3'])
normalize_and_plot(close,['ALSO3', 'B3SA3', 'CARD3'])
normalize_and_plot(close,['BEEF3', 'BRAP4', 'CAML3', 'CMIG3'])
normalize_and_plot(close,['ALPA4', 'ALSO3', 'BKBR3', 'CAML3'])
normalize_and_plot(close,['ALSO3', 'B3SA3', 'BBSE3'])
normalize_and_plot(close,['BPAN4', 'BRPR3', 'CESP6'])
normalize_and_plot(close,['ABCB4', 'BRSR6', 'CSMG3'])
normalize_and_plot(close,['ATOM3', 'ELET3', 'ELET6'])


normalize_and_plot(close,['MULT3', 'POMO3', 'RAIL3', 'RAPT4'])

res2 = DataFrame(results,columns=['rank','lag','tickers']).sort_values(by=['rank','lag'], ascending=[False, True])
# res2[res2['rank']==2]
ind = 60
tickers = res2['tickers'].loc[ind]
normalize_and_plot(close,tickers)


select_order(close,['CRFB3', 'CSAN3', 'CXCE11B', 'DIRR3', 'ECOR3'],maxlags=12)


normalize_and_plot(close,['BBSE3','ALPA4','ALSO3'])
normalize_and_plot(close,['ARZZ3','ALPA4','ALSO3'])
normalize_and_plot(close,['ANIM3','BRML3'])
normalize_and_plot(close,['CPFE3','CSMG3'])
normalize_and_plot(close,['COCE5','VIVT4'])
normalize_and_plot(close,['CPFE3', 'CPLE6', 'CRFB3', 'CTNM4', 'CYRE3'])
normalize_and_plot(close,['CPFE3', 'CPLE6', 'CRFB3', 'DIRR3', 'ECOR3'])
normalize_and_plot(close,['CRFB3', 'CSAN3', 'CXCE11B', 'DIRR3', 'ECOR3'])
normalize_and_plot(close,['TIET4', 'TIMP3', 'TPIS3', 'WEGE3'])
normalize_and_plot(close,['TCSA3', 'TGMA3', 'TIET4', 'YDUQ3'])
normalize_and_plot(close,['TCSA3', 'TEND3', 'TIET4', 'TIMP3'])
normalize_and_plot(close,['TIMP3', 'TRIS3', 'USIM5', 'WEGE3'])
normalize_and_plot(close,['TECN3', 'TIET3', 'TIET4', 'WEGE3'])
normalize_and_plot(close,['TIMP3', 'TOTS3', 'VIVR3', 'WEGE3'])
normalize_and_plot(close,['TCSA3', 'TECN3', 'TRIS3', 'UCAS3'])
normalize_and_plot(close,['AGRO3', 'ALPA4', 'AZUL4', 'BBDC3'])
normalize_and_plot(close,['BBAS3', 'BBDC4', 'BBFI11B', 'BBRK3'])
normalize_and_plot(close,['ABCB4', 'BBAS3', 'BBDC4', 'APER3'])
normalize_and_plot(close,['UNIP6','VIVT4'])
normalize_and_plot(close,['VALE3','VULC3'])
normalize_and_plot(close,['USIM5','VIVR3'])
normalize_and_plot(close,['BBAS3','BBDC4','BBRK3'])
normalize_and_plot(close,['ABCB4', 'BBAS3', 'BBDC4', 'BBRK3'])
normalize_and_plot(close,['ABCB4', 'BBAS3', 'BBDC4', 'APER3'])
normalize_and_plot(close,['ARZZ3', 'BBRK3', 'CARD3', 'CMIG4'])






new_close = close[[col for col in close if 'BBDC' in col]]
new_close = close[['VALE3','PETR4']]
new_close = np.log(close[[col for col in close if 'B' in col]])
new_close = np.log(close[['ITUB4','VALE3','PETR4','BBAS3','CSNA3']])
new_close = np.log(close[['ITUB4','BBAS3','BBDC4','BBDC3','ITUB3','ITSA4']])
new_close = np.log(close.iloc[:,20:25])

results = select_coint_rank(endog = new_close, det_order = 0, k_ar_diff = 1)
print(results.summary())
print(results.rank)
print(results.neqs)
results2 = coint_johansen(endog = new_close, det_order = 0, k_ar_diff = 1)
print(results2.evec)
