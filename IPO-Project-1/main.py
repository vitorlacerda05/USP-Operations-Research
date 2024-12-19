import pandas as pd
import numpy as np
from geopy.distance import geodesic
import pulp
import folium

# Função para definir fatores multiplicadores de custo


def get_tipo_cano(ponto_i, ponto_j):
    tipo_i = 'SAAE' if 'SAAE' in ponto_i else 'Universidade'
    tipo_j = 'SAAE' if 'SAAE' in ponto_j else 'Universidade'
    if tipo_i == 'SAAE' and tipo_j == 'SAAE':
        return 0.63  # SAAE - SAAE
    elif (tipo_i == 'SAAE' and tipo_j == 'Universidade') or (tipo_i == 'Universidade' and tipo_j == 'SAAE'):
        return 0.59  # SAAE - Universidade or Universidade - SAAE
    elif tipo_i == 'Universidade' and tipo_j == 'Universidade':
        return 0.74  # Universidade - Universidade
    else:
        return None  # Outros casos não considerados


# Dados dos pontos (Tabela 1)
data = {
    'Ponto': ['SAAE 1', 'SAAE 2', 'SAAE 3', 'SAAE 4', 'SAAE 5', 'SAAE 6', 'SAAE 7', 'SAAE 8',
              'Campus 1 USP', 'Campus 2 USP', 'UFSCar'],
    'Tipo': ['Suprimento']*8 + ['Demanda']*3,
    'Suprimento': [30, 40, 50, 30, 50, 30, 30, 90, -120, -100, -130]
}
df_pontos = pd.DataFrame(data)

# Coordenadas geográficas (Tabela 2)
coordenadas = {
    'Ponto': ['SAAE 1', 'SAAE 2', 'SAAE 3', 'SAAE 4', 'SAAE 5', 'SAAE 6', 'SAAE 7', 'SAAE 8',
              'Campus 1 USP', 'Campus 2 USP', 'UFSCar'],
    'Latitude': [-22.056067, -22.029408, -22.022213, -22.009414, -22.005740, -22.007573, -22.006214, -21.989030,
                 -22.008018, -22.008774, -21.983547],
    'Longitude': [-47.906624, -47.874192, -47.897600, -47.892042, -47.873665, -47.889043, -47.889628, -47.917174,
                  -47.896600, -47.929775, -47.880959]
}
df_coords = pd.DataFrame(coordenadas)

# Mesclar os DataFrames
df_pontos = df_pontos.merge(df_coords, on='Ponto')

# Inicializar um DataFrame vazio para armazenar as distâncias
distancias = pd.DataFrame(index=df_pontos['Ponto'], columns=df_pontos['Ponto'])

# Calcular as distâncias usando a fórmula de Haversine via geopy
for i, row_i in df_pontos.iterrows():
    for j, row_j in df_pontos.iterrows():
        coord_i = (row_i['Latitude'], row_i['Longitude'])
        coord_j = (row_j['Latitude'], row_j['Longitude'])
        if row_i['Ponto'] != row_j['Ponto']:  # Apenas calcular distâncias entre pontos distintos
            distancia = geodesic(coord_i, coord_j).kilometers
            distancias.at[row_i['Ponto'], row_j['Ponto']] = distancia
        else:
            # Distância de um ponto para ele mesmo é 0
            distancias.at[row_i['Ponto'], row_j['Ponto']] = 0

# Inicializar um DataFrame vazio para os custos
custos = pd.DataFrame(index=df_pontos['Ponto'], columns=df_pontos['Ponto'])

# Calcular os custos
for i, row_i in df_pontos.iterrows():
    for j, row_j in df_pontos.iterrows():
        tipo_cano = get_tipo_cano(row_i['Ponto'], row_j['Ponto'])
        if tipo_cano is not None and row_i['Ponto'] != row_j['Ponto']:
            custo = tipo_cano * distancias.at[row_i['Ponto'], row_j['Ponto']]
            custos.at[row_i['Ponto'], row_j['Ponto']] = custo
        else:
            custos.at[row_i['Ponto'], row_j['Ponto']] = None  # Não há conexão

# Criar o problema de minimização
prob = pulp.LpProblem("Fluxo_de_Agua", pulp.LpMinimize)

# Criar variáveis de decisão
fluxos = {}
for i in df_pontos['Ponto']:
    for j in df_pontos['Ponto']:
        if custos.at[i, j] is not None and i != j:
            var_name = f'x_{i}_{j}'
            fluxos[(i, j)] = pulp.LpVariable(var_name, lowBound=0)

# Função objetivo
prob += pulp.lpSum([fluxos[(i, j)] * float(custos.at[i, j])
                    for (i, j) in fluxos]), "Custo_Total"

# Restrições de conservação de fluxo (ajustadas)
for ponto in df_pontos['Ponto']:
    # Fluxo que entra no ponto
    fluxo_entrada = pulp.lpSum([fluxos[(i, j)]
                               for (i, j) in fluxos if j == ponto])
    # Fluxo que sai do ponto
    fluxo_saida = pulp.lpSum([fluxos[(i, j)]
                             for (i, j) in fluxos if i == ponto])
    # Suprimento/Demanda do ponto
    b_i = df_pontos.loc[df_pontos['Ponto'] == ponto, 'Suprimento'].values[0]
    # Ajuste da restrição
    prob += (fluxo_saida - fluxo_entrada ==
             b_i), f"Conservacao_de_Fluxo_{ponto}"

# Resolver o problema
prob.solve()

# Mostrar o status da solução
print(f"Status: {pulp.LpStatus[prob.status]}")

# Mostrar o valor ótimo da função objetivo
print(f"Custo Total: {pulp.value(prob.objective)}")

# Criar um DataFrame para todos os fluxos, incluindo zeros
fluxos_todos = []
for (i, j), var in fluxos.items():
    valor = var.varValue
    fluxos_todos.append({'Origem': i, 'Destino': j, 'Fluxo': valor})
df_todos_fluxos = pd.DataFrame(fluxos_todos)

# Extrair os fluxos ótimos (fluxos com valor positivo)
df_fluxos = df_todos_fluxos[df_todos_fluxos['Fluxo'] > 0]

# Criar o mapa centrado em São Carlos
mapa = folium.Map(location=[-22.0174, -47.8900], zoom_start=13)

# Adicionar marcadores para os pontos
for _, row in df_pontos.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"{row['Ponto']} ({'Suprimento' if row['Suprimento'] > 0 else 'Demanda'}: {
            abs(row['Suprimento'])})",
        icon=folium.Icon(
            color='green' if row['Tipo'] == 'Suprimento' else 'red')
    ).add_to(mapa)

# Adicionar linhas pretas para todas as conexões possíveis, com popup mostrando o fluxo
for idx, row in df_todos_fluxos.iterrows():
    origem = df_pontos[df_pontos['Ponto'] == row['Origem']].iloc[0]
    destino = df_pontos[df_pontos['Ponto'] == row['Destino']].iloc[0]
    flow_value = row['Fluxo']
    popup_text = f"Fluxo: {flow_value}"
    folium.PolyLine(
        locations=[(origem['Latitude'], origem['Longitude']),
                   (destino['Latitude'], destino['Longitude'])],
        weight=1,
        color='black',
        opacity=0.3,
        popup=popup_text
    ).add_to(mapa)

# Adicionar as linhas azuis representando os fluxos ótimos
for _, row in df_fluxos.iterrows():
    origem = df_pontos[df_pontos['Ponto'] == row['Origem']].iloc[0]
    destino = df_pontos[df_pontos['Ponto'] == row['Destino']].iloc[0]
    # Ajuste da espessura para melhor visualização
    espessura = max(2, row['Fluxo'] / 10)
    folium.PolyLine(
        locations=[(origem['Latitude'], origem['Longitude']),
                   (destino['Latitude'], destino['Longitude'])],
        weight=espessura,
        color='blue',
        opacity=0.8,
        popup=f"Fluxo: {row['Fluxo']}"
    ).add_to(mapa)

# Salvar o mapa
mapa.save('fluxos_agua_SaoCarlos.html')
