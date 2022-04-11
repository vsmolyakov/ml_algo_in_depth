import numpy as np
import string

class GeneticAlgorithm():
    
    def __init__(self, target_string, population_size, mutation_rate):
        self.target = target_string
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.letters = [" "] + list(string.ascii_letters)

    def initialize(self):
        # init population with random strings
        self.population = []
        for _ in range(self.population_size):
            individual = "".join(np.random.choice(self.letters, size=len(self.target)))
            self.population.append(individual)

    def calculate_fitness(self):
        #calculate fitness of each individual in a population
        population_fitness = []
        for individual in self.population:
            # calculate loss as the distance between characters
            loss = 0
            for i in range(len(individual)):
                letter_i1 = self.letters.index(individual[i])
                letter_i2 = self.letters.index(self.target[i])
                loss += abs(letter_i1 - letter_i2)
            fitness = 1 / (loss + 1e-6)
            population_fitness.append(fitness)
        return population_fitness

    def mutate(self, individual):
        #randomly change the characters with probability equal to mutation_rate
        individual = list(individual)
        for j in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                individual[j] = np.random.choice(self.letters)
        return "".join(individual)

    def crossover(self, parent1, parent2):
        #create children from parents by crossover
        cross_i = np.random.randint(0, len(parent1))
        child1 = parent1[:cross_i] + parent2[cross_i:]
        child2 = parent2[:cross_i] + parent1[cross_i:]
        return child1, child2

    def run(self, iterations):
        self.initialize()

        for epoch in range(iterations):
            population_fitness = self.calculate_fitness()
            
            fittest_individual = self.population[np.argmax(population_fitness)]
            highest_fitness = max(population_fitness)

            if fittest_individual == self.target:
                break

            #select individual as a parent proportional to individual's fitness
            parent_probabilities = [fitness / sum(population_fitness) for fitness in population_fitness]

            #next generation
            new_population = []
            for i in np.arange(0, self.population_size, 2):
                #select two parents
                parent1, parent2 = np.random.choice(self.population, size=2, p=parent_probabilities, replace=False)
                #crossover to produce offspring
                child1, child2 = self.crossover(parent1, parent2)
                #save mutated offspring for next generation
                new_population += [self.mutate(child1), self.mutate(child2)]

            print("iter %d, closest candidate: %s, fitness: %.4f" %(epoch, fittest_individual, highest_fitness))
            self.population = new_population
        
        print("iter %d, final candidate: %s" %(epoch, fittest_individual))

if __name__ == "__main__":

    target_string = "Genome"
    population_size = 50
    mutation_rate = 0.1

    ga = GeneticAlgorithm(target_string, population_size, mutation_rate)
    ga.run(iterations = 1000)










