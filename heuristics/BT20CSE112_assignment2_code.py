import heapq
import random


def generate_random_adjacency_matrix(n, min_weight=1, max_weight=10):
    if n <= 0:
        raise ValueError("n must be a positive integer")

    # Initialize an n x n matrix filled with zeros
    adjacency_matrix = [[0] * n for _ in range(n)]

    # Fill in the matrix with random edge weights
    for i in range(n):
        for j in range(i + 1, n):
            # Generate a random integer between min_weight and max_weight
            weight = random.randint(min_weight, max_weight)
            # Set the weight for both directions (since it's an undirected graph)
            adjacency_matrix[i][j] = weight
            adjacency_matrix[j][i] = weight

    return adjacency_matrix


class Element:
    def __init__(self, route, gn, hn, fnc):
        self.route = route  # List of visited indices
        self.gn = gn        # Distance to follow the route
        self.hn = hn        # MST value excluding nodes in the route
        self.fn = gn + hn   # Evaluation function
        self.fnchild = fnc  # useful to add value in child

    def __lt__(self, other):
        # Define comparison for heapq
        return self.fn < other.fn


class DisjointSet:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_x] = root_y
                if self.rank[root_x] == self.rank[root_y]:
                    self.rank[root_y] += 1


def kruskal_mst(adj_matrix, excluded_indices):
    num_nodes = len(adj_matrix)
    edges = []

    # Create a list of edges with weights
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if i not in excluded_indices and j not in excluded_indices:
                edges.append((i, j, adj_matrix[i][j]))

    # Sort edges by weight
    edges.sort(key=lambda x: x[2])

    mst = []
    mst_value = 0
    disjoint_set = DisjointSet(num_nodes)

    # Main loop to build the MST
    while edges:
        u, v, weight = edges.pop(0)
        if disjoint_set.find(u) != disjoint_set.find(v):
            mst.append((u, v, weight))
            mst_value += weight
            disjoint_set.union(u, v)

    return mst_value


def a_star_tsp(adj_matrix):
    num_cities = len(adj_matrix)

    # Calculate the MST value as the initial heuristic
    initial_heuristic = kruskal_mst(adj_matrix, [])
    initial_heuristic_leaving_0 = kruskal_mst(adj_matrix, [0])
    # Initialize the priority queue for A* with Element objects
    start_element = Element([0], 0, initial_heuristic,
                            initial_heuristic_leaving_0)
    priority_queue = [start_element]

    while priority_queue:
        # print_priority_queue(priority_queue)
        element = heapq.heappop(priority_queue)

        route = element.route
        cost = element.gn
        current_city = route[-1] if route else 0

        if len(route) == num_cities + 1:
            return cost, element.route
        elif len(route) == num_cities:
            # all cities are done
            new_route = element.route + [0]
            new_cost = cost + adj_matrix[current_city][0]
            # All cities have been visited, return to the starting city
            new_element = Element(new_route, new_cost, 0, 0)
            heapq.heappush(priority_queue, new_element)
            print(route)
        else:
            for next_city in range(num_cities):
                if next_city not in route:
                    new_route = route + [next_city]
                    new_cost = cost + adj_matrix[current_city][next_city]
                    heuristic_value_to_add_to_child = element.fnchild
                    fnchild = kruskal_mst(adj_matrix, new_route)
                    new_element = Element(
                        new_route, new_cost, heuristic_value_to_add_to_child, fnchild)
                    heapq.heappush(priority_queue, new_element)

    return float('inf')  # No solution found


# Example adjacency matrix for cities
# adjacency_matrix = [[0, 3, 6, 2],
#                     [3, 0, 3, 7],
#                     [6, 3, 0, 3],
#                     [2, 7, 3, 0]]
n = 20
adjacency_matrix = generate_random_adjacency_matrix(n)

# Calculate the TSP using A* with Kruskal's MST as the heuristic
tsp_cost, tsp_route = a_star_tsp(adjacency_matrix)
print("Minimum TSP Cost:", tsp_cost)
print("TSP route ", tsp_route)
