import numpy as np

# Função principal para resolver problemas de programação linear usando o método Simplex de duas fases
def simplex(A, b, c, inequalities=None):
    # Determinar o número de restrições (m) e variáveis (n)
    m, n = A.shape
    num_iterations = 0  # Contador de iterações

    # Se não forem fornecidas desigualdades, assumimos que todas são '<='
    if inequalities is None:
        inequalities = ['<='] * m

    # Ajustar as restrições para garantir que todos os valores em b sejam não negativos
    for i in range(m):
        if b[i] < 0:  # Se b[i] < 0, multiplicar a linha inteira por -1
            A[i, :] *= -1
            b[i] *= -1
            # Ajustar o tipo de desigualdade conforme necessário
            if inequalities[i] == '<=':
                inequalities[i] = '>='
            elif inequalities[i] == '>=':
                inequalities[i] = '<='

    # Inicializar contadores de variáveis adicionais
    total_vars = n  # Total de variáveis (originais + adicionais)
    slack_vars = 0  # Contador de variáveis de folga
    surplus_vars = 0  # Contador de variáveis de excesso
    artificial_vars = 0  # Contador de variáveis artificiais

    # Determinar o tipo de variável adicional para cada desigualdade
    for ineq in inequalities:
        if ineq == '<=':
            slack_vars += 1  # Adiciona uma variável de folga
            total_vars += 1
        elif ineq == '>=':
            surplus_vars += 1  # Adiciona uma variável de excesso
            artificial_vars += 1  # Adiciona também uma variável artificial
            total_vars += 2
        elif ineq == '=':
            artificial_vars += 1  # Apenas variável artificial
            total_vars += 1
        else:
            raise ValueError(f"Inequality type '{ineq}' not recognized.")

    # Criar matriz aumentada A_eq para incluir todas as variáveis adicionais
    A_eq = np.zeros((m, total_vars))
    b_eq = b.copy()  # Copiar vetor de recursos
    c_extended = np.zeros(total_vars)  # Vetor de custos estendido
    base = []  # Lista para rastrear as variáveis na base
    artificial_indices = []  # Índices das variáveis artificiais
    current_var_index = n  # Começar após as variáveis originais

    # Processar as desigualdades e adicionar variáveis adicionais conforme necessário
    for i in range(m):
        # Copiar coeficientes das variáveis originais para a matriz aumentada A_eq
        A_eq[i, :n] = A[i]
        # Recuperar o tipo de desigualdade da restrição atual
        ineq = inequalities[i]

        if ineq == '<=':  
            # Adicionar variável de folga
            A_eq[i, current_var_index] = 1
            base.append(current_var_index)
            current_var_index += 1
        elif ineq == '>=':  
            # Adicionar variável de excesso e variável artificial
            A_eq[i, current_var_index] = -1
            current_var_index += 1
            A_eq[i, current_var_index] = 1
            base.append(current_var_index)
            artificial_indices.append(current_var_index)
            current_var_index += 1
        elif ineq == '=':  
            # Adicionar apenas variável artificial
            A_eq[i, current_var_index] = 1
            base.append(current_var_index)
            artificial_indices.append(current_var_index)
            current_var_index += 1
        else:
            raise ValueError(f"Inequality type '{ineq}' not recognized.")

    # Atualizar os custos das variáveis originais no vetor estendido
    c_extended[:n] = c

    # Criar o tableau inicial
    tableau = np.zeros((m + 1, total_vars + 1))
    tableau[:m, :-1] = A_eq  # Coeficientes das restrições
    tableau[:m, -1] = b_eq  # Vetor de recursos (lado direito)

    # Configurar função objetivo para a Fase I, se houver variáveis artificiais
    if artificial_indices:
        c_phase1 = np.zeros(total_vars)  # Vetor de custos para a Fase I
        c_phase1[artificial_indices] = 1  # Custos das variáveis artificiais são 1 na Fase I
        tableau[-1, :-1] = c_phase1  # Atualizar a linha da função objetivo no tableau

        # Ajustar o tableau para eliminar custos artificiais das variáveis básicas
        for idx in artificial_indices:
            # Para cada variável artificial na base, ajustar a função objetivo
            row = np.where(tableau[:m, idx] == 1)[0][0]
            tableau[-1, :] -= tableau[row, :]

    max_iteracoes = 1000  # Limite de iterações para evitar loops infinitos
    status = None  # Inicializar status da solução

    # Função auxiliar para determinar a variável entrante usando regra do custo reduzido mínimo
    def encontrar_variavel_entrante(custos):
        min_value = np.min(custos)
        if min_value >= -1e-8:  # Nenhum custo negativo, solução básica ótima
            return None
        return np.argmin(custos)  # Índice da variável com menor custo reduzido

    # Fase I: Encontrar solução básica viável
    if artificial_indices:
        while True:
            num_iterations += 1
            custos = tableau[-1, :-1]  # Custos reduzidos
            entra = encontrar_variavel_entrante(custos)
            if entra is None:  # Solução básica viável encontrada
                break

            # Determinar a variável a sair usando razão mínima
            coluna = tableau[:m, entra]
            rhs = tableau[:m, -1]  # Recursos
            razoes = np.full(m, np.inf)
            indices_positivos = np.where(coluna > 1e-8)[0]
            if indices_positivos.size == 0:  # Nenhuma variável pode sair, solução ilimitada
                status = 'ILIMITADA'
                return status, None, None, num_iterations

            razoes[indices_positivos] = rhs[indices_positivos] / coluna[indices_positivos]
            sai = indices_positivos[np.argmin(razoes[indices_positivos])]

            # Pivoteamento
            pivo = tableau[sai, entra]
            tableau[sai, :] /= pivo  # Normalizar linha pivô
            for i in range(m + 1):
                if i != sai:
                    tableau[i, :] -= tableau[i, entra] * tableau[sai, :]

            base[sai] = entra

            if num_iterations >= max_iteracoes:  # Verificar limite de iterações
                status = 'ILIMITADA'
                return status, None, None, num_iterations

        # Verificar viabilidade após a Fase I
        if abs(tableau[-1, -1]) > 1e-8:
            status = 'INFACTÍVEL'  # Problema sem solução viável
            return status, None, None, num_iterations

        # Ajustar tableau e custos para a Fase II (remover variáveis artificiais)
        tableau = np.delete(tableau, artificial_indices, axis=1)
        c_extended = np.delete(c_extended, artificial_indices)

    # Fase II: Otimização da função objetivo original
    c_extended = np.hstack([c_extended, 0])
    tableau[-1, :] = -c_extended[base].dot(tableau[:-1, :]) + c_extended

    while True:
        num_iterations += 1
        custos = tableau[-1, :-1]
        entra = encontrar_variavel_entrante(custos)
        if entra is None:  # Solução ótima encontrada
            status = 'ÓTIMA'
            break

        coluna = tableau[:m, entra]
        rhs = tableau[:m, -1]
        razoes = np.full(m, np.inf)
        indices_positivos = np.where(coluna > 1e-8)[0]
        if indices_positivos.size == 0:
            status = 'ILIMITADA'
            return status, None, None, num_iterations

        razoes[indices_positivos] = rhs[indices_positivos] / coluna[indices_positivos]
        sai = indices_positivos[np.argmin(razoes[indices_positivos])]

        # Pivoteamento
        pivo = tableau[sai, entra]
        tableau[sai, :] /= pivo
        for i in range(m + 1):
            if i != sai:
                tableau[i, :] -= tableau[i, entra] * tableau[sai, :]

        base[sai] = entra

        if num_iterations >= max_iteracoes:
            status = 'ILIMITADA'
            return status, None, None, num_iterations

    # Extrair solução ótima
    x_opt = np.zeros(total_vars)
    for i in range(m):
        x_opt[base[i]] = tableau[i, -1]

    x_opt_original = x_opt[:n]  # Variáveis originais
    valor_otimo = c @ x_opt_original  # Valor ótimo da função objetivo

    return status, x_opt_original, valor_otimo, num_iterations            

# Teste do algoritmo com as instâncias fornecidas

# Problema 1: tem que ter solução ótima
c1 = np.array([-1, -4, -6, -9, -4, -3, -4, -5, -1, -3, -1, -6, -5,
               -9, -1, -8, -7, -6, -5, -7, -7, -7, -1, -2, -4, -2, -3, -3, -2, -2])
A1 = np.array([[7, 4, 8, 3, 9, 5, 8, 3, 1, 1, 7, 1, 4, 6, 7, 9, 4, 8, 6, 6, 8, 3, 7, 8, 6, 5, 4, 5, 6, 2],
               [5, 3, 1, 4, 5, 4, 7, 4, 2, 2, 6, 2, 9, 1, 5, 2,
                8, 8, 4, 8, 2, 5, 3, 9, 1, 2, 5, 2, 1, 9],
               [4, 5, 1, 4, 1, 7, 2, 2, 2, 7, 8, 1, 4, 4, 2, 9,
                2, 2, 8, 7, 5, 8, 8, 1, 8, 4, 3, 7, 5, 6],
               [2, 8, 6, 2, 6, 9, 2, 3, 6, 8, 6, 9, 3, 2, 3, 3,
                1, 5, 5, 8, 4, 9, 8, 6, 4, 1, 7, 7, 8, 7],
               [2, 4, 6, 9, 4, 4, 7, 6, 7, 7, 4, 5, 5, 4, 3, 9,
                9, 9, 6, 3, 8, 3, 3, 6, 5, 7, 2, 3, 9, 5],
               [6, 7, 5, 1, 5, 8, 8, 6, 1, 9, 7, 8, 2, 4, 4, 5,
                3, 5, 2, 2, 5, 3, 4, 7, 8, 7, 3, 7, 7, 5],
               [9, 8, 2, 3, 5, 2, 3, 2, 9, 2, 3, 9, 2, 4, 8, 5,
                5, 1, 3, 5, 4, 6, 1, 8, 8, 3, 3, 3, 2, 1],
               [8, 2, 8, 2, 8, 9, 7, 6, 1, 4, 9, 4, 3, 1, 8, 2,
                4, 7, 9, 5, 8, 5, 5, 6, 7, 6, 4, 6, 7, 3],
               [3, 1, 9, 5, 4, 4, 5, 2, 3, 7, 1, 4, 1, 7, 6, 6,
                7, 3, 2, 5, 9, 8, 5, 4, 2, 1, 1, 3, 5, 2],
               [4, 9, 8, 3, 8, 9, 7, 6, 1, 8, 9, 5, 6, 2, 1, 5, 5, 2, 8, 6, 9, 1, 2, 8, 6, 6, 7, 1, 7, 2]])
b1 = np.array([43, 41, 28, 37, 17, 33, 35, 22, 24, 49])

# Definir as desigualdades
inequalities1 = ['<='] * len(b1)

# Resolver o Problema 1
status1, x1, optimal_value1, num_iterations1 = simplex(A1, b1, c1, inequalities1)
print("Problema 1:")
print("Status:", status1)
if x1 is not None:
    print("Solução Ótima (x):", x1)
else:
    print("Solução Ótima (x): None")
print("Valor Ótimo da Função Objetivo:", optimal_value1)
print("Número de Iterações:", num_iterations1)
print("\n")

# Problema 2: é ilimitada
c2 = np.array([-8, -1, -5, -7, -6, -8, -8, -1, -3, -2, -5, -1, -9,
               -8, -4, -1, -4, -9, -6, -6, -8, -9, -8, -1, -2, -7, -5, -2, -5, -5])
A2 = np.array([[-7, 2, 3, 3, 4, 5, 7, 9, 9, 9, 5, 5, 3, 9, 3, 4, 9, 8, 5, 5, 3, 9, 1, 2, 8, 5, 9, 1, 3, 2],
               [-1, 2, 5, 1, 3, 7, 4, 9, 9, 3, 2, 7, 4, 3, 4,
                8, 7, 4, 7, 6, 9, 7, 3, 3, 5, 5, 5, 7, 2, 5],
               [-9, 3, 7, 4, 4, 3, 5, 5, 5, 4, 4, 5, 6, 7, 3,
                5, 6, 6, 8, 8, 3, 6, 9, 9, 9, 6, 2, 6, 3, 1],
               [-5, 7, 8, 1, 9, 1, 4, 8, 8, 9, 9, 9, 4, 1, 2,
                3, 1, 6, 3, 5, 7, 7, 6, 8, 5, 9, 4, 9, 4, 1],
               [-5, 9, 9, 3, 2, 9, 1, 6, 3, 7, 9, 9, 7, 7, 2,
                5, 3, 7, 6, 4, 9, 9, 9, 8, 5, 4, 1, 1, 2, 7],
               [-8, 7, 2, 2, 7, 4, 2, 1, 4, 2, 2, 1, 2, 4, 8,
                2, 7, 6, 4, 1, 2, 6, 5, 4, 2, 5, 6, 9, 3, 7],
               [-1, 4, 3, 1, 9, 6, 5, 4, 2, 5, 8, 5, 3, 1, 6,
                7, 1, 5, 1, 8, 5, 7, 8, 5, 7, 7, 3, 2, 9, 7],
               [-7, 4, 2, 6, 4, 2, 8, 6, 4, 7, 4, 4, 4, 3, 8,
                2, 5, 5, 7, 1, 7, 4, 3, 3, 7, 5, 3, 2, 5, 5],
               [-3, 9, 5, 3, 1, 7, 9, 3, 6, 3, 7, 4, 6, 3, 8,
                3, 3, 3, 1, 5, 4, 7, 6, 4, 7, 4, 8, 9, 7, 5],
               [-1, 1, 5, 7, 7, 6, 2, 3, 4, 7, 7, 9, 8, 6, 3, 6, 1, 1, 2, 1, 5, 8, 9, 9, 9, 9, 2, 1, 5, 8]])
b2 = np.array([35, 16, 11, 37, 34, 17, 48, 41, 19, 33])

# Definir as desigualdades
inequalities2 = ['<='] * len(b2)

# Ajustar A2 e b2 para garantir que b2 é não negativo
for i in range(b2.shape[0]):
    if b2[i] < 0:
        A2[i, :] *= -1
        b2[i] *= -1
        if inequalities2[i] == '<=':
            inequalities2[i] = '>='
        elif inequalities2[i] == '>=':
            inequalities2[i] = '<='

# Resolver o Problema 2
status2, x2, optimal_value2, num_iterations2 = simplex(A2, b2, c2, inequalities2)
print("Problema 2:")
print("Status:", status2)
if x2 is not None:
    print("Solução Ótima (x):", x2)
else:
    print("Solução Ótima (x): None")
print("Valor Ótimo da Função Objetivo:", optimal_value2)
print("Número de Iterações:", num_iterations2)
print("\n")

# Problema 3: é infactível
c3 = np.array([-4, -2, -1, -3, -3, -8, -7, -7, -7, -7, -3, -4, -5,
               -1, -9, -5, -1, -6, -1, -1, -5, -3, -2, -3, -7, -6, -8, -7, -8, -8])
A3 = np.array([[7, 4, 4, 7, 2, 1, 9, 2, 3, 9, 1, 7, 3, 5, 1, 6, 1, 7, 3, 2, 1, 7, 2, 2, 1, 7, 7, 8, 5, 3],
               [9, 2, 9, 4, 4, 2, 2, 7, 5, 3, 7, 6, 5, 6, 7, 1,
                5, 8, 1, 9, 6, 7, 1, 4, 7, 5, 9, 5, 4, 3],
               [4, 1, 8, 2, 4, 9, 5, 5, 8, 2, 4, 3, 3, 6, 1, 2,
                6, 4, 8, 3, 3, 1, 8, 7, 2, 8, 2, 3, 1, 6],
               [9, 3, 3, 7, 3, 7, 1, 5, 5, 5, 4, 2, 6, 1, 1, 2,
                4, 8, 9, 2, 8, 9, 2, 2, 9, 2, 6, 9, 3, 4],
               [2, 2, 5, 7, 9, 6, 5, 8, 7, 6, 3, 9, 6, 5, 3, 1,
                2, 5, 6, 3, 6, 2, 9, 7, 3, 7, 9, 6, 4, 2],
               [2, 4, 1, 7, 7, 3, 2, 1, 2, 9, 6, 3, 1, 7, 1, 5,
                1, 9, 1, 7, 1, 1, 1, 5, 1, 6, 6, 6, 1, 9],
               [4, 2, 5, 4, 9, 8, 8, 3, 4, 5, 4, 7, 9, 2, 4, 1,
                9, 7, 8, 2, 8, 8, 9, 7, 1, 1, 5, 2, 6, 2],
               [8, 1, 4, 5, 1, 9, 5, 1, 4, 6, 7, 6, 5, 7, 6, 4,
                2, 5, 3, 2, 7, 7, 6, 5, 6, 4, 5, 6, 8, 1],
               [1, 1, 5, 2, 2, 9, 6, 2, 2, 4, 6, 8, 3, 2, 2, 1,
                7, 3, 4, 1, 1, 2, 2, 1, 4, 5, 3, 2, 7, 7],
               [4, 3, 6, 4, 6, 7, 9, 1, 1, 1, 1, 8, 1, 4, 7, 7, 6, 5, 5, 7, 5, 8, 9, 9, 3, 8, 6, 2, 8, 6]])
b3 = np.array([-1, 43, 12, 13, 31, 11, 47, 15, 13, 43])

# Definir as desigualdades
inequalities3 = ['<='] * len(b3)

# Garantir que b3 é não negativo
for i in range(b3.shape[0]):
    if b3[i] < 0:
        A3[i, :] *= -1
        b3[i] *= -1
        if inequalities3[i] == '<=':
            inequalities3[i] = '>='
        elif inequalities3[i] == '>=':
            inequalities3[i] = '<='

# Resolver o Problema 3
status3, x3, optimal_value3, num_iterations3 = simplex(A3, b3, c3, inequalities3)
print("Problema 3:")
print("Status:", status3)
if x3 is not None:
    print("Solução Ótima (x):", x3)
else:
    print("Solução Ótima (x): None")
print("Valor Ótimo da Função Objetivo:", optimal_value3)
print("Número de Iterações:", num_iterations3)
