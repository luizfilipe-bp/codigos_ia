# Códigos de Inteligência Artificial

Este repositório contém uma coleção de códigos desenvolvidos para a disciplina de **Inteligência Artificial** na Universidade Federal de Lavras. Cada diretório representa um projeto ou exercício diferente, explorando diversos algoritmos e conceitos da área.

---

## Algoritmos Implementados

### 1. K-Nearest Neighbors (KNN)
**Descrição:**  
O KNN é um algoritmo de aprendizado supervisionado utilizado para **classificação** e **regressão**. Ele classifica um novo ponto de dado com base na classe majoritária de seus 'k' vizinhos mais próximos.

**Implementações:**
- `main_hardcode.py`: Implementação do KNN a partir do zero, utilizando distância Euclidiana para encontrar vizinhos mais próximos e classificação por votação majoritária. O código utiliza o dataset Iris, dividindo-o em dados de treinamento e teste.
- `main_biblioteca.py`: Implementação utilizando a biblioteca **scikit-learn** aplicada aos datasets **Iris** e **Wine**, avaliando acurácia, precisão e revocação para diferentes valores de 'k'.

**Dataset:**
- `Iris.csv`: Dados de três espécies de flores Iris (**Setosa**, **Versicolor** e **Virginica**), com medições de sépalas e pétalas.

---

### 2. K-Means
**Descrição:**  
O K-Means é um algoritmo de aprendizado **não supervisionado** utilizado para **agrupamento (clustering)**. O objetivo é particionar 'n' observações em 'k' clusters, nos quais cada observação pertence ao cluster com a média mais próxima.

**Implementações:**
- `main.py`: Implementação do K-Means a partir do zero. Inicializa os centróides, atribui cada ponto ao centróide mais próximo (passo de expectativa) e recalcula os centróides (passo de maximização).
- `main_sklearn.py`: Implementação utilizando **scikit-learn**, explorando também **PCA** para redução de dimensionalidade e visualização dos clusters.

**Dataset:**
- `Iris.csv`: O mesmo dataset utilizado no KNN.

---

### 3. Algoritmo Genético
**Descrição:**  
Algoritmo de otimização inspirado no processo de **seleção natural**, utilizando população, seleção, **crossover** e **mutação** para encontrar soluções ótimas ou aproximadas para problemas complexos.

**Implementação:**
- `main.py`: Implementa um algoritmo genético para encontrar o valor máximo de uma função matemática. Inclui representação binária de indivíduos, função de fitness, seleção por torneio, crossover de um ponto e mutação bit a bit.

---

### 4. Métodos de Busca (DFS e A*)
**Descrição:**
- **DFS (Busca em Profundidade):** Explora o mais longe possível ao longo de cada ramo antes de retroceder.
- **A* (A-estrela):** Busca informada que utiliza heurística para encontrar o caminho de menor custo entre um nó inicial e um final.

**Implementação:**
- `main.py`: Resolve o **Jogo dos Oito (8-puzzle)** utilizando busca em profundidade iterativa e o algoritmo A*. A heurística utilizada para o A* é a **distância Euclidiana**.

---

### 5. Redes Neurais Artificiais (RNA)
**Descrição:**  
Modelo computacional inspirado no cérebro humano. RNAs são compostas por **neurônios artificiais interconectados** e são capazes de aprender e reconhecer padrões.

**Implementação:**
- `main.py`: Utiliza a classe **MLPClassifier** da **scikit-learn** para treinar uma rede neural para classificação. Aplicada aos datasets **Iris** e **Wine**, com pré-processamento utilizando **StandardScaler** para normalização.

---
