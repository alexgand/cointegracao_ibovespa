# antigo, pra mais de duas series (johansen test):

data_folder = 'C:/Users/alega/Documents/Mestrado Stats/Séries Temporais/artigo/data/'
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
# ano_inicio = '2010-01'
ano_inicio = '2018-06'
ano_fim = '2018-12'
for ticker in base:
    base[ticker] = base[ticker][ano_inicio:ano_fim]

#deleta $PTAX:
if '$PTAX' in base:
    del base['$PTAX']

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


k = 4
combinations = list(itertools.combinations(close.columns[15:30], k))
# combinations = list(itertools.combinations(close.columns[:], k))

print('quantidade de combinacoes de',str(k),'tickers:',len(combinations))

results = []

for comb in tqdm(combinations):
	# new_close = np.log(close[[item for item in comb]]) # em logs
	new_close = (close[[item for item in comb]])

	lags = select_order(new_close.values,maxlags=12)
	best_lag = lags.selected_orders['aic']

	if best_lag >= 0:
	# if best_lag == 1:

		# select_rank = select_coint_rank(endog = new_close, det_order = -1, k_ar_diff = 1, signif = sign_level)
		# select_rank = select_coint_rank(endog = new_close, det_order = -1, k_ar_diff = best_lag, signif = sign_level)
		select_rank = select_coint_rank(endog = new_close, det_order = 0, k_ar_diff = best_lag, signif = sign_level)
		# select_rank = select_coint_rank(endog = new_close, det_order = 0, k_ar_diff = best_lag, signif = 0.1)
		
		if (select_rank.rank > 0) and (select_rank.rank < select_rank.neqs): # rank cheio e rank 0 nao interessam
			results.append((select_rank.rank, best_lag, [item for item in comb]))
			# print((select_rank.rank, [item for item in comb]))

print(results)
print(len(results))
results_df = DataFrame(results,columns=['rank','lag','tickers']).sort_values(by=['rank','lag'], ascending=[False, True])
print(results_df.head(50))

############ de novo agora com outro periodo:

data_folder = 'C:/Users/alega/Documents/Mestrado Stats/Séries Temporais/artigo/data/'
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
# ano_inicio = '2010-01'
ano_inicio = '2019-01'
ano_fim = '2019-06'
for ticker in base:
    base[ticker] = base[ticker][ano_inicio:ano_fim]

#deleta $PTAX:
if '$PTAX' in base:
    del base['$PTAX']

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




combinations = results_df['tickers'] # pega os resultados do periodo anterior

results = []

for comb in tqdm(combinations):
	# new_close = np.log(close[[item for item in comb]]) # em logs
	
	if all([True if item in close else False for item in comb]): # se todos estao na nova base

		new_close = (close[[item for item in comb]])

		lags = select_order(new_close.values,maxlags=12)
		best_lag = lags.selected_orders['aic']

		if best_lag >= 0:
		# if best_lag == 1:

			# select_rank = select_coint_rank(endog = new_close, det_order = -1, k_ar_diff = 1, signif = sign_level)
			# select_rank = select_coint_rank(endog = new_close, det_order = -1, k_ar_diff = best_lag, signif = sign_level)
			select_rank = select_coint_rank(endog = new_close, det_order = 0, k_ar_diff = best_lag, signif = sign_level)
			# select_rank = select_coint_rank(endog = new_close, det_order = 0, k_ar_diff = best_lag, signif = 0.1)
			
			if (select_rank.rank > 0) and (select_rank.rank < select_rank.neqs): # rank cheio e rank 0 nao interessam
				results.append((select_rank.rank, best_lag, [item for item in comb]))
				# print((select_rank.rank, [item for item in comb]))

print(results)
print(len(results))
results_df2 = DataFrame(results,columns=['rank','lag','tickers']).sort_values(by=['rank','lag'], ascending=[False, True])
print(results_df2.head(50))


########### novo periodo:

data_folder = 'C:/Users/alega/Documents/Mestrado Stats/Séries Temporais/artigo/data/'
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
# ano_inicio = '2010-01'
ano_inicio = '2019-06'
ano_fim = '2019-12'
for ticker in base:
    base[ticker] = base[ticker][ano_inicio:ano_fim]

#deleta $PTAX:
if '$PTAX' in base:
    del base['$PTAX']

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

combinations = results_df2['tickers'] # pega os resultados do periodo anterior

results = []

for comb in tqdm(combinations):
	# new_close = np.log(close[[item for item in comb]]) # em logs
	
	if all([True if item in close else False for item in comb]): # se todos estao na nova base

		new_close = (close[[item for item in comb]])

		lags = select_order(new_close.values,maxlags=12)
		best_lag = lags.selected_orders['aic']

		if best_lag >= 0:
		# if best_lag == 1:

			# select_rank = select_coint_rank(endog = new_close, det_order = -1, k_ar_diff = 1, signif = sign_level)
			# select_rank = select_coint_rank(endog = new_close, det_order = -1, k_ar_diff = best_lag, signif = sign_level)
			select_rank = select_coint_rank(endog = new_close, det_order = 0, k_ar_diff = best_lag, signif = sign_level)
			# select_rank = select_coint_rank(endog = new_close, det_order = 0, k_ar_diff = best_lag, signif = 0.1)
			
			if (select_rank.rank > 0) and (select_rank.rank < select_rank.neqs): # rank cheio e rank 0 nao interessam
				results.append((select_rank.rank, best_lag, [item for item in comb]))
				# print((select_rank.rank, [item for item in comb]))

print(results)
print(len(results))
results_df3 = DataFrame(results,columns=['rank','lag','tickers']).sort_values(by=['rank','lag'], ascending=[False, True])
print(results_df3.head(50))



##### PRINTS em TODO O PERIODO:

data_folder = 'C:/Users/alega/Documents/Mestrado Stats/Séries Temporais/artigo/data/'
arquivos = descobre_arquivos_na_pasta(data_folder,tipo_do_arquivo='.csv')
base = {}
arquivos = descobre_arquivos_na_pasta(data_folder,tipo_do_arquivo='.csv')
for arquivo in tqdm(arquivos):
    base[arquivo[:-4]] = pd.read_csv(data_folder + arquivo, parse_dates=['Date'],index_col='Date')
    base[arquivo[:-4]]['Adj Close'] = base[arquivo[:-4]]['Close']
    base[arquivo[:-4]] = base[arquivo[:-4]][['High','Open','Low','Close','Volume','Adj Close','Open Interest']]

#reduz tamanho da base, comeca em determinado ano:
ano_inicio = '2018-06'
ano_fim = '2019-12'
for ticker in base:
    base[ticker] = base[ticker][ano_inicio:ano_fim]

pd.options.mode.chained_assignment = None  # default='warn'

ticker_mais_antigo = 'PETR4'

# dfs close:
close = DataFrame(0,columns=base.keys(),index=base[ticker_mais_antigo].index)
for ticker in tqdm(base):
    close[ticker] = base[ticker]['Close']

# retira todos os ativos que nao tiveram negocio (nan) em algum dia dos dez anos - tanto os que comecaram depois quanto os que foram extintos antes.
close = close.dropna(axis=1)

normalize_and_plot(close,results_df3.sample()['tickers'].iloc[0])
normalize_and_plot(close,['BEES3', 'BIDI4', 'BPAN4', 'BRDT3'])
