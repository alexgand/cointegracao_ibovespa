# Autor: Alexandre Gandini
# fev/2020
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

data_folder = 'C:/Users/alega/Documents/Mestrado_stats/series_temp/artigo/data/'
arquivos = descobre_arquivos_na_pasta(data_folder,tipo_do_arquivo='.csv')
base = {}
arquivos = descobre_arquivos_na_pasta(data_folder,tipo_do_arquivo='.csv')
for arquivo in tqdm(arquivos):
    base[arquivo[:-4]] = pd.read_csv(data_folder + arquivo, parse_dates=['Date'],index_col='Date')
    base[arquivo[:-4]]['Adj Close'] = base[arquivo[:-4]]['Close']
    base[arquivo[:-4]] = base[arquivo[:-4]][['High','Open','Low','Close','Volume','Adj Close','Open Interest']]

#reduz tamanho da base, comeca em determinado ano:
ano_inicio = '20050101'
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

#retira os tickers com dois numeros no final, fundos imobiliarios, etc.:
tickers_excluir = []
for ticker in base:
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

k = 2
combinations = list(itertools.combinations(close.columns[:], k))

print('quantidade de combinacoes de',str(k),'tickers:',len(combinations))

##############
# testes de estacionariedade das series:
##############

results_estacionariedade = []

for ticker in tqdm(close.columns):

	adf = adfuller(np.log(close[ticker]))
	
	if adf[1] <= sign_level: #there is a NOT unit root, sao estacionárias.

		results_estacionariedade.append((adf[0],adf[1], adf[2], ticker))

results_df_estacionariedade = DataFrame(results_estacionariedade,columns=['adf','p_value','usedlag','ticker']).sort_values(by=['adf','usedlag'], ascending=[True, True])
print(results_df_estacionariedade.head(50))

# agora diferenciando em um:

results_estacionariedade = []

for ticker in tqdm(close.columns):

	adf = adfuller(np.log(close[ticker]).diff().dropna())
	
	if adf[1] <= sign_level: #there is a NOT unit root, sao estacionárias.

		results_estacionariedade.append((adf[0],adf[1], adf[2], ticker))

results_df_estacionariedade = DataFrame(results_estacionariedade,columns=['adf','p_value','usedlag','ticker']).sort_values(by=['adf','usedlag'], ascending=[True, True])
print(results_df_estacionariedade.head(50))

# null: there is a unit root, não são estacionárias - resultados: agora diferenciando elas, todas as séries ficam estacionarias - sao integradas de ordem um.

##############

# com janelas moveis, ver quem tem mais janelas de cointegracao.

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

# exportacao de tabela pra ler no latex (tem que retirar os 'brancos' do primeiro multiindex):
qtd_primeiros_a_mostrar = 15
agrupados_count['ticker1'] = agrupados_count.index.get_level_values(1)
agrupados_count = agrupados_count.droplevel(level=1)
agrupados_count = agrupados_count[['ticker1','adf']]
agrupados_count.columns = ['Ticker 2','Quantidade de janelas']
agrupados_count.index.name = 'Ticker 1'
agrupados_count.head(qtd_primeiros_a_mostrar).to_csv('C:/Users/alega/Documents/Mestrado_stats/series_temp/artigo/artigo_latex/qtd_janelas.csv')

#random plot:
sample = melhores.sample().index[0]
# escolher tickers plot:
ticker0 = 'ABEV3'
ticker1 = 'UGPA3'
sample = [ticker0, ticker1]
ax = normalize_and_plot(close,list(sample)) # plot normal
plt.ylabel('log das cotações iniciando em 1')
plt.xlabel('')
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

ticker0 = 'CCRO3'
ticker1 = 'POMO4'

ticker0 = 'VIVT4'
ticker1 = 'WEGE3'