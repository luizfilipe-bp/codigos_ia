"""
    Trabalho Prático 05 - Algoritmo Genético
    Nome: Luiz Filipe Bartelega Penha
"""
import random


class GeneticAlgorithm:
    def __init__(self, number_of_individuals=4, generations=5, mutation_rate=0.01, crossover_rate=0.7):
        self.number_of_individuals = number_of_individuals
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.lowerBound = -10
        self.upperBound = 10
        self.population = []
        self.number_of_bits = 20

    def float_to_binary(self, x):
        normalized = (x - self.lowerBound) / (self.upperBound - self.lowerBound)
        individual_normalized = int(normalized * (2 ** self.number_of_bits - 1))
        return format(individual_normalized, f'0{self.number_of_bits}b')

    def binary_to_float(self, x):
        int_val = int(x, 2)
        normalized = int_val / (2 ** self.number_of_bits - 1)
        return round(self.lowerBound + normalized * (self.upperBound - self.lowerBound), 6)

    def fitness_function(self, x):
        x_float = self.binary_to_float(x)
        return x_float ** 2 - 3 * x_float + 4

    def initialize_population(self):
        for i in range(self.number_of_individuals):
            individual = round(random.uniform(self.lowerBound, self.upperBound), 6)
            binary_individual = self.float_to_binary(individual)
            self.population.append(binary_individual)

    def tournament(self):
        population_copy = list(self.population)
        parents = []

        for i in range(2):
            positions = len(population_copy) - 1
            first_parent_position = random.randint(0, positions)
            parent1 = population_copy[first_parent_position]

            second_parent_position = random.randint(0, positions)
            while first_parent_position == second_parent_position:
                second_parent_position = random.randint(0, positions)
            parent2 = population_copy[second_parent_position]

            if self.fitness_function(parent1) > self.fitness_function(parent2):
                parents.append(population_copy.pop(first_parent_position))
            else:
                parents.append(population_copy.pop(second_parent_position))

        return parents[0], parents[1]

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, self.number_of_bits - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            return child1, child2
        return parent1, parent2

    def mutation(self, individual):
        individual = list(individual)
        for i in range(self.number_of_bits):
            if random.random() < self.mutation_rate:
                if individual[i] == '0':
                    individual[i] = '1'
                if individual[i] == '1':
                    individual[i] = '0'
        return ''.join(individual)

    def best_individual(self):
        best_individual = self.population[0]
        best_fitness = self.fitness_function(best_individual)
        for individual in self.population:
            fitness = self.fitness_function(individual)
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual
        best_float = self.binary_to_float(best_individual)
        return best_individual, best_float, best_fitness

    def print_best_individual(self):
        best_individual, best_float, best_fitness = self.best_individual()
        print(f'Melhor indivíduo: {best_individual} ({best_float}) com fitness: {best_fitness:.5f}')
        with open(f"resultadosInd{self.number_of_individuals}gen{self.generations}.txt", "a") as file:
            best_fitness = round(best_fitness, 5)
            best_fitness = str(best_fitness).replace('.', ',')
            file.write(f'{best_fitness}\n')

    def run(self):
        self.initialize_population()

        for generation in range(self.generations):
            new_generation = []
            while len(new_generation) < self.number_of_individuals:
                parent1, parent2 = self.tournament()
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)

                new_generation.append(child1)
                if len(new_generation) < self.number_of_individuals:
                    new_generation.append(child2)

            self.population = new_generation
        self.print_best_individual()


if __name__ == '__main__':
    ag = GeneticAlgorithm()
    ag.run()
