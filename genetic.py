import numpy as np
import random
import multiprocessing as mp
import csv
import os
from tqdm import tqdm
import sys
from Agents.RandomAgent import RandomAgent as ra
from Agents.AdrianHerasAgent import AdrianHerasAgent as aha
from Agents.AlexPastorAgent import AlexPastorAgent as apa
from Agents.AlexPelochoJaimeAgent import AlexPelochoJaimeAgent as apja
from Agents.CarlesZaidaAgent import CarlesZaidaAgent as cza
from Agents.CrabisaAgent import CrabisaAgent as ca
from Agents.EdoAgent import EdoAgent as ea
from Agents.PabloAleixAlexAgent import PabloAleixAlexAgent as paaa
from Agents.SigmaAgent import SigmaAgent as sa
from Agents.TristanAgent import TristanAgent as ta
from Managers.GameDirector import GameDirector
import sys


AGENTS = [ra, aha, apa, apja, cza, ca, ea, paaa, sa, ta]


def create_individual(n):
    """ Crear un individuo: array con floats que entre todos sumen 1"""
    prob = [random.random() for _ in range(n)]
    ind = [p / sum(prob) for p in prob]  # Normalización

    return ind

def create_diverse_individual(n):
    """Crear un individuo con valores más diversos, algunos pequeños y otros grandes, normalizados."""
    prob = [0] * n
    for i in range(n):
        if random.random() < 0.8:  # 80% de probabilidad de usar cada método
            prob[i] = random.random() 
        else:
            prob[i] = random.uniform(10, 1000) 

    ind = [p / sum(prob) for p in prob]  # Normalización

    return ind

def create_population(n, m):
    """Crear una población de individuos generados con valores aleatorios normalizados."""
    return [create_individual(m) for _ in range(n)]

def create_diverse_population(n, m):
    """Crear una población de individuos con mayor diversidad en los valores."""
    return [create_diverse_individual(m) for _ in range(n)]

def save_population(population, filename):
    """Guardar la población en un archivo CSV."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(population)

def load_population(filename):
    """Cargar la población desde un archivo CSV."""
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        return [[float(value) for value in row] for row in reader]

def initialize_population(size_population, size_individual, pop_filename):
    """Inicializar una población cargándola de un archivo o creándola si no existe."""
    if os.path.exists(pop_filename):
        return load_population(pop_filename)
    else:
        population = create_population(size_population, size_individual)
        save_population(population, pop_filename)
        return population
    
def initialize_diverse_population(size_population, size_individual, pop_filename):
    """Inicializar una población diversa cargándola de un archivo o creándola si no existe."""
    if os.path.exists(pop_filename):
        return load_population(pop_filename)
    else:
        population = create_diverse_population(size_population, size_individual)
        save_population(population, pop_filename)
        return population

def chose_agent(ind):
    """Elegir un agente aleatoriamente con una probabilidad dada por los pesos del individuo."""
    return random.choices(AGENTS, weights=ind, k=1)[0]

def play_game(game_id, ind):
    """Simular una partida entre un agente seleccionado y tres oponentes diferentes a él, devolviendo su rendimiento (fitness)."""
    random.seed(game_id)
    fitness = 0
    agent = chose_agent(ind)

   # Elegir 3 agentes que pueden repetirse, excluyendo el agente principal
    opponents = random.choices([a for a in AGENTS if a != agent], k=3)
    
    # Crear lista de jugadores y mezclar el orden
    players = [agent] + opponents
    random.shuffle(players)
    
    # Inicializar la partida
    game_director = GameDirector(agents=players, max_rounds=200,  store_trace=False)
    game_trace = game_director.game_start(game_id, False)
    
    last_round = max(game_trace["game"].keys(), key=lambda r: int(r.split("_")[-1]))
    last_turn = max(game_trace["game"][last_round].keys(), key=lambda t: int(t.split("_")[-1].lstrip("P")))
    victory_points = game_trace["game"][last_round][last_turn]["end_turn"]["victory_points"]
    winner = max(victory_points, key=lambda player: int(victory_points[player]))
    winner_pos = int(winner.lstrip("J"))
    if agent == players[winner_pos]:
            fitness += 1

    return fitness

def generate_games(n_games, ind):
    """Jugar múltiples partidas en paralelo para evaluar el rendimiento promedio de un individuo."""
    with mp.Pool(processes=mp.cpu_count()-1) as pool:
        fitness = pool.starmap(play_game, [(i, ind) for i in range(n_games)])
    total_fitness = sum(fitness)
    return total_fitness/n_games

def fitness(ind, n_games):
    """Calcular la aptitud de un individuo en función de su desempeño en múltiples juegos."""
    fitness = generate_games(n_games, ind)
    return fitness


def tournament_selection(population, fitness_values, tournament_size=3):
    """Selecciona dos individuos usando selección por torneo."""   
    tournament1 = random.sample(list(zip(population, fitness_values)), tournament_size)
    winner1 = max(tournament1, key=lambda x: x[1])  
    tournament2 = random.sample(list(zip(population, fitness_values)), tournament_size)
    winner2 = max(tournament2, key=lambda x: x[1])     
    return [winner1[0], winner2[0]]


def roulette_selection(population, fitness_values):
    """Selecciona dos individuos usando selección por ruleta."""    
    total_fitness = sum(fitness_values)    
    if total_fitness == 0:
        parents = random.sample(population, 2)  
    else:
        probabilities = [f / total_fitness for f in fitness_values]
        # Seleccionar dos individuos con las probabilidades calculadas
        parents = random.choices(population, weights=probabilities, k=2)        
    return parents

def elitism_selection(population, fitness_values, elite_size=2):
    """Selecciona los mejores individuos directamente (elitismo)."""
    sorted_population = sorted(zip(population, fitness_values), key=lambda x: x[1], reverse=True)
    elite = [ind[0] for ind in sorted_population[:elite_size]]
    return elite

def crossover_one_point(parent1, parent2):
    """Realiza un cruce de un solo punto aleatorio entre dos padres y asegura que los hijos sean consistentes (suma = 1)."""
    
    # Escoger un punto de cruce aleatorio
    crossover_point = random.randint(1, len(parent1) - 1)    
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]

    def normalize(child):
        total = sum(child)
        return [gene / total for gene in child]
    child1 = normalize(child1)
    child2 = normalize(child2)
    
    return [child1, child2]

def crossover_two_point(parent1, parent2):
    """Realiza un cruce de dos puntos aleatorios entre dos padres y asegura que los hijos sean consistentes (suma = 1)."""
    
    # Escoger dos puntos de cruce aleatorios
    point1 = random.randint(1, len(parent1) - 2)
    point2 = random.randint(point1 + 1, len(parent1) - 1)
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    
    def normalize(child):
        total = sum(child)
        return [gene / total for gene in child]
    
    child1 = normalize(child1)
    child2 = normalize(child2)
    
    return [child1, child2]


def mutate(child, mutation_prob):
    """
    Realiza una mutación aleatoria con una probabilidad de mutación, con 2 opciones:
    - intercambiando 2 elementos del array del hijo.
    - sumando a un elemento aleatorio un incremento y después normalizamos.
    """
    
    if random.random() > mutation_prob:
        return child  # Retornamos el hijo sin cambios
    
    if random.random() > 0.5: #intercambio
        indices = random.sample(range(len(child)), 2)
        child[indices[0]], child[indices[1]] = child[indices[1]], child[indices[0]]
        return child
    else: #incremento a un elemento aleatorio
        indice = random.sample(range(len(child)), 1)
        incr = random.uniform(0.05, 0.2)
        child[indice[0]] += incr
        child = [p/sum(child) for p in child]
        return child


def stationary_replacement(population, fitness_population, children, fitness_children):
    """
    Reemplazar los peores individuos de la población con los hijos generados. Es decir, si los hijos son los peores también se reemplazan.
    Se asegura de que la población mantenga el mismo tamaño.
    """
    combined_population = population + children
    combined_fitness = fitness_population + fitness_children
    sorted_population = sorted(zip(combined_population, combined_fitness), key=lambda x: x[1], reverse=True)
    new_population = [ind[0] for ind in sorted_population[:len(population)]]
    new_fitness = [ind[1] for ind in sorted_population[:len(population)]]

    return new_population, new_fitness



def run_experiment(size_population, max_generations, prob_mutation, n_games, selection_function, initialize_function, crossover_function, output_file, pop_filename):
    """Ejecutar un experimento con los parámetros dados y guardar los resultados en un archivo CSV."""

    print(f"Configuración del experimento: Población={size_population}, Generaciones={max_generations}, Mutación={prob_mutation}, Juegos={n_games}, Selección={selection_function.__name__}, Cruce={crossover_function.__name__}, Inicialización={initialize_function.__name__}")

    size_individual = 10
    population = initialize_function(size_population, size_individual, pop_filename)   
    fitness_population = [fitness(ind, n_games) for ind in population]
    best_overall_fitness = float('-inf')


    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "Max Fitness", "Avg Fitness", "Best Individual"])
        for generation in tqdm(range(max_generations), desc=f"Exp: {output_file}", file=sys.stderr):
            selected_parents = selection_function(population, fitness_population)
            children = crossover_function(selected_parents[0], selected_parents[1])
            mutated_child1 = mutate(children[0], prob_mutation)
            mutated_child2 = mutate(children[1], prob_mutation)

            fitness_children = [fitness(mutated_child1, n_games), fitness(mutated_child2, n_games)]
            population, fitness_population = stationary_replacement(population, fitness_population, [mutated_child1, mutated_child2], fitness_children)

            best_fitness = max(fitness_population)
            avg_fitness = sum(fitness_population) / len(fitness_population)
            best_individual = population[fitness_population.index(best_fitness)]
            print(f"---> Gen: {generation}, Best Fitness: {best_fitness}, Avg Fitness: {avg_fitness}, Mejor individuo: {best_individual}")
            writer.writerow([generation, best_fitness, avg_fitness, best_individual])

            # Actualizar el mejor individuo global
            if best_fitness > best_overall_fitness:
                best_overall_fitness = best_fitness
                best_overall_individual = best_individual
            
    print(f"Evolución del fitness guardada en {output_file}")
    print(f"Mejor individuo encontrado: {best_overall_individual} con fitness {best_overall_fitness}")


# Diccionarios de funciones y parámetros
POPULATION_INITIALIZERS = {
    "normal": initialize_population,
    "diverse": initialize_diverse_population
}

SELECTION_FUNCTIONS = {
    "tournament": tournament_selection,
    "roulette": roulette_selection,
    "elitism": elitism_selection
}

CROSSOVER_FUNCTIONS = {
    "onepoint": crossover_one_point,
    "twopoint": crossover_two_point,
}

# Variaciones de parámetros
POPULATION_SIZES = [15, 100]
MAX_GENERATIONS = [100, 200]
MUTATION_PROBABILITIES = [0.2, 0.8]
N_GAMES = [50, 100]
SELECTION_METHODS = list(SELECTION_FUNCTIONS.keys())
CROSSOVER_TYPES = list(CROSSOVER_FUNCTIONS.keys())
POPULATION_TYPES = list(POPULATION_INITIALIZERS.keys())



if __name__ == "__main__":

    # RESUMEN DE LOS 9 EXPERIMENTOS MÁS COMPLETOS DE LOS 35 QUE SE HAN REALIZADO

    pop_filename = f"initial_diverse_population_size15.csv"
    filename = f"results_csv/results_diverse_pop15_gen100_mut0.2_games50_selelitism_cruce2.csv"
    run_experiment(15, 100, 0.2, 50, SELECTION_FUNCTIONS["elitism"], POPULATION_INITIALIZERS["diverse"], CROSSOVER_FUNCTIONS["twopoint"], filename, pop_filename)

    pop_filename = f"initial_diverse_population_size15.csv"
    filename = f"results_csv/results_diverse_pop15_gen100_mut0.8_games50_selelitism_cruce2.csv"
    run_experiment(15, 100, 0.8, 50, SELECTION_FUNCTIONS["elitism"], POPULATION_INITIALIZERS["diverse"], CROSSOVER_FUNCTIONS["twopoint"], filename, pop_filename)

    pop_filename = f"initial_diverse_population_size15.csv"
    filename = f"results_csv/results_diverse_pop15_gen100_mut0.8_games50_selroulette_cruce2.csv"
    run_experiment(15, 100, 0.8, 50, SELECTION_FUNCTIONS["roulette"], POPULATION_INITIALIZERS["diverse"], CROSSOVER_FUNCTIONS["twopoint"], filename, pop_filename)

    pop_filename = f"initial_diverse_population_size15.csv"
    filename = f"results_csv/results_diverse_pop15_gen100_mut0.8_games50_seltournament_cruce1.csv"
    run_experiment(15, 100, 0.8, 50, SELECTION_FUNCTIONS["tournament"], POPULATION_INITIALIZERS["diverse"], CROSSOVER_FUNCTIONS["onepoint"], filename, pop_filename)

    pop_filename = f"initial_diverse_population_size15.csv"
    filename = f"results_csv/results_diverse_pop15_gen200_mut0.8_games50_selelitism_cruce2.csv"
    run_experiment(15, 200, 0.8, 100, SELECTION_FUNCTIONS["elitism"], POPULATION_INITIALIZERS["diverse"], CROSSOVER_FUNCTIONS["twopoint"], filename, pop_filename)

    pop_filename = f"initial_diverse_population_size100.csv"
    filename = f"results_csv/results_diverse_pop100_gen100_mut0.8_games50_selelitism_cruce2.csv"
    run_experiment(100, 100, 0.8, 50, SELECTION_FUNCTIONS["elitism"], POPULATION_INITIALIZERS["diverse"], CROSSOVER_FUNCTIONS["twopoint"], filename, pop_filename)

    pop_filename = f"initial_diverse_population_size100.csv"
    filename = f"results_csv/results_diverse_pop100_gen100_mut0.8_games100_selelitism_cruce2.csv"
    run_experiment(100, 100, 0.8, 100, SELECTION_FUNCTIONS["elitism"], POPULATION_INITIALIZERS["diverse"], CROSSOVER_FUNCTIONS["twopoint"], filename, pop_filename)

    pop_filename = f"initial_normal_population_size15.csv"
    filename = f"results_csv/results_normal_pop15_gen100_mut0.8_games50_selelitism_cruce2.csv"
    run_experiment(15, 100, 0.8, 100, SELECTION_FUNCTIONS["elitism"], POPULATION_INITIALIZERS["normal"], CROSSOVER_FUNCTIONS["twopoint"], filename, pop_filename)

    pop_filename = f"initial_normal_population_size100.csv"
    filename = f"results_csv/results_normal_pop100_gen100_mut0.8_games50_selelitism_cruce2.csv"
    run_experiment(100, 100, 0.8, 100, SELECTION_FUNCTIONS["elitism"], POPULATION_INITIALIZERS["normal"], CROSSOVER_FUNCTIONS["twopoint"], filename, pop_filename)
