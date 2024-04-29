import time
from functools import partial
from collections import namedtuple
from typing import List, Callable, Tuple
from random import choices, randint, random

Genome = List[int]
Population = List[Genome]
FitnessFunc = Callable[[Genome], int]
PopulateFunc = Callable[[], Population]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]

num_cities = 51
# Generate field names using a list comprehension
field_names = [f'city_{i}' for i in range(1, num_cities + 1)]

# Define the named tuple class "Thing"
Thing = namedtuple('Thing', field_names)


def generate_random_adjacency_matrix(n, min_weight=1, max_weight=10):
    if n <= 0:
        raise ValueError("n must be a positive integer")

    # Initialize an n x n matrix filled with zeros
    adjacency_matrix = [[0] * n for _ in range(n)]

    # Fill in the matrix with random edge weights
    for i in range(n):
        for j in range(i + 1, n):
            # Generate a random integer between min_weight and max_weight
            weight = randint(min_weight, max_weight)
            # Set the weight for both directions (since it's an undirected graph)
            adjacency_matrix[i][j] = weight
            adjacency_matrix[j][i] = weight

    return adjacency_matrix


adjacency_matrix = generate_random_adjacency_matrix(num_cities)
things = []
for row in adjacency_matrix:
    thing = Thing(*row)
    things.append(thing)


def generate_genome(length: int) -> Genome:
    gnome = [0]
    while True:
        if len(gnome) == length:
            gnome.append(gnome[0])
            break

        temp = randint(1, length-1)
        if temp not in gnome:
            gnome.append(temp)
    return gnome


def generate_population(size: int, genome_length: int) -> Population:
    return [generate_genome(genome_length) for _ in range(size)]


def fitness(genome: Genome, things: List[Thing], weight_limit: int = 100000) -> float:
    if len(genome) != len(things) + 1:
        print(len(genome))
        print(len(things))
        # raise ValueError("genome and things must be of the same length")
        raise ValueError(
            "genome and things must be such that len(genome) should be equal to len(things)+1")

    weight = 0
    value = 0
    for i, element in enumerate(genome):
        if (i == len(genome)-1):
            value = (1/weight)*10.0*(len(genome)-1)
            return value
        else:

            weight = weight + things[element][genome[i+1]]


def selection_pair(population: Population, fitness_func: FitnessFunc) -> Tuple[Genome, Genome]:
    return tuple(choices(
        population=population,
        weights=[fitness_func(genome) for genome in population],
        k=2
    ))


def single_point_crossover(a: List[int], b: List[int]) -> Tuple[List[int], List[int]]:
    if len(a) != len(b):
        raise ValueError("Genome a and b must be of the same length")

    # Ensure the genomes have a length greater than or equal to 2
    length = len(a)
    if length < 2:
        return a, b

    # Choose a random crossover point
    while True:
        posa = randint(1, length - 2)
        print(posa)
        posb = randint(1, length - 2)
        print(posb)
        if (posa != posb):
            break
    # Determine the start and end positions for the crossover
    start_position = min(posa, posb)
    end_position = max(posa, posb)

    '''Goal: Keep elements from start_position to end_position as it is as a in offspring_a and fill the other
      positions by elements of b in order from i=0 to i=b.length -1. all would be distinct'''
    reserved_list = a[start_position:(end_position+1)]
    print(reserved_list)
    offspring_a = []
    to_be_filled = b.copy()
    print(to_be_filled)
    for element in reserved_list:
        to_be_filled.remove(element)
    j = 0
    for i in range(len(a)):
        if (i >= start_position and i <= end_position):
            offspring_a.append(a[i])
        else:
            offspring_a.append(to_be_filled[j])
            j = j+1
    while True:
        posa = randint(1, length - 2)
        print(posa)
        posb = randint(1, length - 2)
        print(posb)
        if (posa != posb):
            break
    start_position = min(posa, posb)
    end_position = max(posa, posb)

    '''Goal: Keep elements from start_position to end_position as it is as a in offspring_a and fill the other
      positions by elements of b in order from i=0 to i=b.length -1. all would be distinct'''
    reserved_list = b[start_position:(end_position+1)]
    print(reserved_list)
    offspring_b = []
    to_be_filled = a.copy()
    print(to_be_filled)
    for element in reserved_list:
        to_be_filled.remove(element)
    j = 0
    for i in range(len(a)):
        if (i >= start_position and i <= end_position):
            offspring_b.append(b[i])
        else:
            offspring_b.append(to_be_filled[j])
            j = j+1
    return offspring_a, offspring_b


def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(num):
        index = randint(1, len(genome) - 2)
        index1 = randint(1, len(genome) - 2)
        if random() > probability:
            genome[index] = genome[index]
        else:
            temp = genome[index1]
            genome[index1] = genome[index]
            genome[index] = temp
    return genome


def run_evolution(
    populate_func: PopulateFunc,
    fitness_func: FitnessFunc,
    fitness_limit: float,
    selection_func: SelectionFunc = selection_pair,
    crossover_func: CrossoverFunc = single_point_crossover,
    mutation_func: MutationFunc = mutation,
    generation_limit: int = 100
) -> Tuple[Population, int]:
    population = populate_func()

    for i in range(generation_limit):
        population = sorted(
            population,
            key=lambda genome: fitness_func(genome, things, weight_limit),
            reverse=True
        )
        if fitness_func(population[0], things, weight_limit) >= fitness_limit:
            break

        next_generation = population[0:3]

        for j in range((len(population) // 2) - 1):
            parents = selection_func(
                population, partial(fitness_func, things=things))
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]

        population = next_generation

    population = sorted(
        population,
        key=lambda genome: fitness_func(genome, things, weight_limit),
        reverse=True
    )

    return population, i


start = time.time()
# # Set the weight limit for the knapsack
weight_limit = 3000
population, generations = run_evolution(
    populate_func=partial(generate_population, size=28,
                          genome_length=len(things)),
    fitness_func=fitness,
    fitness_limit=3,
    generation_limit=200
)
end = time.time()


def genome_to_things(genome: Genome, things: List[Thing]) -> List[str]:
    result = population[0]
    weight = 0
    for (i, element) in enumerate(genome):
        if i == len(genome)-1:
            break
        weight = weight + things[element][genome[i+1]]
    print(f'cost = {weight}')
    return result


print(f"number of generations: {generations}")
print(f"best solution: {genome_to_things(population[0], things)}")
print(f"time: {end-start}s")
