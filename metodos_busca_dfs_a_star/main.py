"""
    Trabalho Prático 04 - Jogo dos Oito (Métodos de Busca)
    Nome: Luiz Filipe Bartelega Penha
"""
import time
import heapq


# Encontra a posição do zero na matriz
def get_empty_space_position(matrix):
    for i in range(3):
        for j in range(3):
            if matrix[i][j] == 0:
                return i, j


# Entrada dos valores da matriz
def input_matrix():
    matrix = []
    print("A célula vazia deve ser indicada como zero!")
    row = 0
    while row < 3:
        row_values = input(f"Digite os valores da linha {row + 1} (separados por espaços): ").split()
        if len(row_values) != 3:
            print("Cada linha deve possuir exatamente 3 valores.")
            print("Digite os valores novamente!\n")
        else:
            row_values = [int(value) for value in row_values]
            matrix.append(row_values)
            row += 1
    return matrix


# Visita os vizinhos do espaço vazio e troca de posição com eles criando novos estados
def get_empty_space_neighbors(current_state):
    empty_space_moves = [(-1, 0),   # up
                         (1, 0),    # down
                         (0, -1),   # left
                         (0, 1)]    # right

    empty_space_row, empty_space_column = get_empty_space_position(current_state)

    neighbors = []
    for move_x, move_y in empty_space_moves:
        new_row = empty_space_row + move_x
        new_column = empty_space_column + move_y

        if 0 <= new_row < 3 and 0 <= new_column < 3:
            new_state = [row[:] for row in current_state]

            # troca o espaço vazio com seu vizinho
            new_state[empty_space_row][empty_space_column], new_state[new_row][new_column] = \
                new_state[new_row][new_column], new_state[empty_space_row][empty_space_column]
            # Novos estados após a troca do empty_space com seus são adicinados a lista de vizinhos do DFS
            neighbors.append(new_state)
    return neighbors


def dfs(actual_state, goal_state, visiteds, depth, max_depth):
    if actual_state == goal_state:
        return True, [actual_state]

    visiteds.append(actual_state)

    if depth > max_depth:
        return False, []

    for neighbor in get_empty_space_neighbors(actual_state):
        if neighbor not in visiteds:
            solved, path = dfs(neighbor, goal_state, visiteds, depth + 1, max_depth)
            if solved:
                return True, [actual_state] + path
    return False, []


def iterative_dfs(initial_state, goal_state, max_depth=50):
    for depth in range(max_depth):
        visiteds = []
        solved, path = dfs(initial_state, goal_state, visiteds, 0, depth)
        if solved:
            return path
    return []


def print_path(path):
    if len(path) != 0:
        print(f"{len(path) - 1} movimentos para chegar ao objetivo")
        print("--- Passos para chegar ao objetivo ---")
        for i, state in enumerate(path):
            if i == 0:
                print("--- Início ---")
            else:
                print(f"Passo {i}")
                if i == len(path) - 1:
                    print("--- Fim --- ")

            for row in state:
                print(f'|{row[0]}|{row[1]}|{row[2]}|')

            print()

    else:
        print("Não foi possível encontrar uma solução para este jogo")
        print()


def get_euclidian_distance(actual_state, final_state):
    positions = {}
    for i in range(3):
        for j in range(3):
            cell = final_state[i][j]
            positions[cell] = (i, j)

    euclidian_distance = 0
    for i in range(3):
        for j in range(3):
            actual_state_cell = actual_state[i][j]
            if actual_state_cell != 0:
                final_state_i, final_state_j = positions[actual_state_cell]
                euclidian_distance += ((i - final_state_i) ** 2 + (j - final_state_j) ** 2) ** 0.5

    return euclidian_distance


def a_star_search(initial_state, final_state):
    closed_list = []
    open_list = []

    g = 0
    h = get_euclidian_distance(initial_state, final_state)
    f = g + h
    heapq.heappush(open_list, (f, g, initial_state, []))

    while open_list:
        f, g, current_state, path = heapq.heappop(open_list)

        if current_state == final_state:
            return path + [current_state]

        closed_list.append(current_state)

        for neighbor in get_empty_space_neighbors(current_state):
            if neighbor not in closed_list:
                new_g = g + 1
                new_h = get_euclidian_distance(neighbor, final_state)
                new_f = new_g + new_h
                heapq.heappush(open_list, (new_f, new_g, neighbor, path + [current_state]))


def print_results(path, final_time, begin_time):
    print(f'Tempo de execução: {(final_time - begin_time):.7f} segundos')
    print_path(path)


if __name__ == '__main__':
    print('** Resolvendo o jogo dos * com métodos de busca A* e Busca em profundidade ** \n')
    print("--- Digite os valores da matriz 3x3 inicial do jogo dos 8---")
    initial_matrix = input_matrix()
    print("\n--- Digite os valores da matriz 3x3 final do jogo dos 8 ---")
    goal_matrix = input_matrix()
    print()

    print('--- Resolvendo o jogo dos 8 utilizando o algoritmo A-estrela (A*) ---')
    begin = time.perf_counter()
    path_to_objective = a_star_search(initial_matrix, goal_matrix)
    final = time.perf_counter()
    print_results(path_to_objective, final, begin)

    print('--- Resolvendo o jogo dos 8 utilizando o algoritmo de busca em profundidade (DFS) ---')
    print('A busca em profundidade é menos eficiente, pode demorar alguns segundos dependendo da instância do problema')
    begin = time.perf_counter()
    path_to_objective = iterative_dfs(initial_matrix, goal_matrix)
    final = time.perf_counter()
    print_results(path_to_objective, final, begin)
