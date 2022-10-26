# Course: CS261 - Data Structures
# Author: Sebastian Gajardo
# Assignment: Assignment 6 Graphs
# Description: Directed weighted Graph implementation using adjacency matrix with add vertex, add edge, remove edge, get
# vertices, get edges, is valid path, depth first search, breadth first search, has cycle and Dijkstra methods.

import heapq
from collections import deque


class DirectedGraph:
    """
    Class to implement directed weighted graph
    - duplicate edges not allowed
    - loops not allowed
    - only positive edge weights
    - vertex names are integers
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency matrix
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.v_count = 0
        self.adj_matrix = []

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            v_count = 0
            for u, v, _ in start_edges:
                v_count = max(v_count, u, v)
            for _ in range(v_count + 1):
                self.add_vertex()
            for u, v, weight in start_edges:
                self.add_edge(u, v, weight)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        if self.v_count == 0:
            return 'EMPTY GRAPH\n'
        out = '   |'
        out += ' '.join(['{:2}'.format(i) for i in range(self.v_count)]) + '\n'
        out += '-' * (self.v_count * 3 + 3) + '\n'
        for i in range(self.v_count):
            row = self.adj_matrix[i]
            out += '{:2} |'.format(i)
            out += ' '.join(['{:2}'.format(w) for w in row]) + '\n'
        out = f"GRAPH ({self.v_count} vertices):\n{out}"
        return out

    # ------------------------------------------------------------------ #

    def add_vertex(self) -> int:
        """
        Adds a new vertex to the graph vertex and returns the number of vertices in the graph after the addition. Name
        doesn't need to be provided instead vertex is assigned a reference index. O(|v|^2).
        """
        for vertex in self.adj_matrix:  # Add a column to existing graph to account for vertex to be added.
            vertex.append(0)

        self.v_count += 1  # Add one to vertex count.
        self.adj_matrix.append([0 for _ in range(self.v_count)])  # Add vertex to graph.

        return self.v_count  # Return new number of vertices.

    def add_edge(self, src: int, dst: int, weight=1) -> None:
        """
        Receives source indices , destination indices and weight and adds a new edge to the graph. If either or both
        indices do not exist in the graph, the weight is not positive or the source and destination refer to the same
        index it does nothing. If an edge already exist, it updates the edge's weight. O(1).
        """
        # Vertices must be in graph, weight positive and source not equal to destination.
        if -1 < src < self.v_count and -1 < dst < self.v_count and weight > 0 and src != dst:
            self.adj_matrix[src][dst] = weight  # Create new edge or update weight.

    def remove_edge(self, src: int, dst: int) -> None:
        """
        Receives a source and destination vertices and removes the edge between them. If either or both vertices do not
        exist in the graph or the edge does not exist it does nothing. O(1).
        """
        # Vertices must be in graph and edge must exist.
        if -1 < src < self.v_count and -1 < dst < self.v_count and self.adj_matrix[src][dst] != 0:
            self.adj_matrix[src][dst] = 0  # Remove edge.

    def get_vertices(self) -> []:
        """
        Returns a list of vertices in the graph, order of the vertices in the list does not matter. O(|v|).
        """
        return [n for n in range(self.v_count)]  # Return list of vertices in the graph.

    def get_edges(self) -> []:
        """
        Returns the list of edges in the graph, each edge is represented bya tuple of source, destination and weight.
        The order of the edges in the list does not matter. O(|v|^2).
        """
        all_edges = []  # Initialize list to hold edge tuples.

        for source in range(self.v_count):
            for destination in range(self.v_count):
                if self.adj_matrix[source][destination] != 0:  # If edge exist add to all edges list in tuple form.
                    all_edges.append((source, destination, self.adj_matrix[source][destination]))

        return all_edges  # Return list of tuple edges.

    def is_valid_path(self, path: []) -> bool:
        """
        Receives a list of vertices and returns True if sequence of vertices is a valid path in the graph. An empty path
        is considered a valid path. With a valid path you can travel from the first vertex to the last vertex at each
        step traversing over an edge. O(n -1)
        """
        if len(path) == 1 and not -1 < path[0] < self.v_count:  # If single vertex and not in graph return False.
            return False

        valid = True  # Valid path boolean.
        n = 0  # Counter to step through path list.

        while valid is True and n < (len(path) - 1) and -1 < path[n] < self.v_count and -1 < path[n + 1] < self.v_count:
            # If valid and not to end of path and both vertices in graph.
            if self.adj_matrix[path[n]][path[n + 1]] == 0:  # If that edge does not exist change valid to False.
                valid = False

            n += 1  # Add one to counter.

        return valid  # Return valid boolean.

    def dfs(self, v_start, v_end=None) -> []:
        """
        Receives the vertex at which to start and an optional parameter of the vertex at which to end (default of None).
        Then performs a depth first search and returns the list of vertices in the order they are visited, if starting
        vertex not in graph returns empty list and if name of end vertex provided but not in graph search is done as if
        there is no end vertex. Vertices explored in ascending order. O(|v| + |e|).
        """
        visited = []  # List of visited vertexes.
        stack = deque([v_start])  # Deque to be used as a stack.

        if not -1 < v_start < self.v_count:  # If starting vertex not in graph return empty list.
            return visited

        while len(stack) != 0:  # While stack not empty, pop top of stack.
            vertex = stack.pop()
            if vertex not in visited:  # If first time encountering vertex add to list.
                visited.append(vertex)

                if vertex == v_end:
                    # If appended vertex is target vertex return visited.
                    return visited

                for neighbor in range(self.v_count - 1, -1, -1):  # Append neighbors to stack in reverse sorted order.
                    if self.adj_matrix[vertex][neighbor] != 0:
                        stack.append(neighbor)

        return visited  # Return vertices in visited order.

    def bfs(self, v_start, v_end=None) -> []:
        """
        Receives the vertex at which to start and an optional parameter of the vertex at which to end (default of None).
        Then performs a breadth first search and returns the list of vertices in the order they are visited, if starting
        vertex not in graph returns empty list and if name of end vertex provided but not in graph search is done as if
        there is no end vertex. Vertices explored in ascending order. O(|v| + |e|).
        """
        visited = []  # List of visited vertexes.
        stack = deque([v_start])  # Deque to be used as a queue.

        if not -1 < v_start < self.v_count:  # If starting vertex not in graph return empty list.
            return visited

        while len(stack) != 0:  # While stack not empty, pop top of stack.
            vertex = stack.popleft()
            if vertex not in visited:  # If first time encountering vertex add to list.
                visited.append(vertex)

                if vertex == v_end:
                    # If appended vertex is target vertex return visited.
                    return visited

                for neighbor in range(self.v_count):  # Append neighbors to stack in reverse sorted order.
                    if self.adj_matrix[vertex][neighbor] != 0:
                        stack.append(neighbor)

        return visited  # Return vertices in visited order.

    def has_cycle(self):
        """
        Return True if there's at least one cycle in the graph, False otherwise. O(|v| + |e|).
        """
        unvisited = self.get_vertices()  # List of unvisited vertices.
        state = {v: "unvisited" for v in range(self.v_count)}  # Dictionary to store vertex states.
        stack = deque()  # Deque to be used as stack.

        while len(unvisited) != 0:  # While all vertices not checked, add first of unvisited to the stack.
            stack.append(unvisited[0])

            while len(stack) != 0:  # While stack not empty get next vertex off top of stack.
                vertex = stack[-1]

                if state[vertex] == "visited":  # If visited pop off top.
                    stack.pop()

                else:  # If not visited.
                    if state[vertex] == "unvisited":  # If unvisited set to exploring and remove from unvisited.
                        state[vertex] = "exploring"
                        unvisited.remove(vertex)

                    continue_path = False  # Start continue path as False.
                    neighbor = self.v_count - 1  # Counter.

                    while neighbor > -1 and continue_path is False:
                        # Append next neighbor to stack if neighbor not in stack, if in stack cycle was found.
                        if self.adj_matrix[vertex][neighbor] != 0 and state[neighbor] != "visited":
                            if neighbor not in stack:
                                stack.append(neighbor)
                                continue_path = True
                            else:  # If in stack it's explored and an ancestor so cycle was found.
                                return True

                        neighbor -= 1  # Reduce counter.

                    if continue_path is False:  # If not continuing path set to visited and pop off stack.
                        state[vertex] = "visited"
                        stack.pop()

        return False  # Return False if no cycle found in graph.

    def dijkstra(self, src: int) -> []:
        """
        Receives a source vertex and computes the shortest length to all other vertices in the graph using a Dijkstra
        algorithm. Returns a list with one value per vertex, if a vertex is not reachable from given vertex the value
        returned is infinity.
        """
        distances = [float("inf") for n in range(self.v_count)]  # Initialize list of distances to infinity.
        distances[src] = 0  # Set distance to source as 0.
        unvisited = self.get_vertices()  # List of unvisited vertices.
        queue = []  # Priority queue to be used.
        heapq.heappush(queue, (0, src))  # Push source value with distance 0 to itself.

        while len(queue) != 0:  # While priority queue not empty keep looking.
            distance, vertex = heapq.heappop(queue)

            if vertex in unvisited:  # If vertex unvisited add distance to distances and remove vertex from unvisited.
                distances[vertex] = distance
                unvisited.remove(vertex)

                for neighbor in range(self.v_count):   # For each reachable neighbor, add to queue.

                    if self.adj_matrix[vertex][neighbor] != 0:  # Distance is the cumulative distance up to that point.
                        heapq.heappush(queue, (distance + self.adj_matrix[vertex][neighbor], neighbor))

        return distances  # Return all obtained distance.


if __name__ == '__main__':

    print("\nPDF - method add_vertex() / add_edge example 1")
    print("----------------------------------------------")
    g = DirectedGraph()
    print(g)
    for _ in range(5):
        g.add_vertex()
    print(g)

    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    for src, dst, weight in edges:
        g.add_edge(src, dst, weight)
    print(g)


    print("\nPDF - method get_edges() example 1")
    print("----------------------------------")
    g = DirectedGraph()
    print(g.get_edges(), g.get_vertices(), sep='\n')
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    print(g.get_edges(), g.get_vertices(), sep='\n')


    print("\nPDF - method is_valid_path() example 1")
    print("--------------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    test_cases = [[0, 1, 4, 3], [1, 3, 2, 1], [0, 4], [4, 0], [], [2]]
    for path in test_cases:
        print(path, g.is_valid_path(path))


    print("\nPDF - method dfs() and bfs() example 1")
    print("--------------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    for start in range(5):
        print(f'{start} DFS:{g.dfs(start)} BFS:{g.bfs(start)}')


    print("\nPDF - method has_cycle() example 1")
    print("----------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)

    edges_to_remove = [(3, 1), (4, 0), (3, 2)]
    for src, dst in edges_to_remove:
        g.remove_edge(src, dst)
        print(g.get_edges(), g.has_cycle(), sep='\n')

    edges_to_add = [(4, 3), (2, 3), (1, 3), (4, 0)]
    for src, dst in edges_to_add:
        g.add_edge(src, dst)
        print(g.get_edges(), g.has_cycle(), sep='\n')
    print('\n', g)

    edges = [(0, 1, 18), (5, 7, 12), (5, 9, 7), (5, 12, 4), (7, 8, 11), (7, 9, 4), (8, 12, 6), (9, 3, 11),
    (9, 6, 18), (9, 8, 12), (11, 1, 6), (12, 10, 2)]
    g = DirectedGraph(edges)
    print(g.has_cycle())

    edges = [(1, 10, 5), (1, 11, 14), (2, 3, 3), (2, 5, 15), (2, 8, 20), (3, 7, 15), (4, 8, 14), (5, 4, 5),
             (7, 4, 10), (7, 6, 9), (9, 1, 19), (11, 2, 20), (12, 10, 12)]
    g = DirectedGraph(edges)
    print(g.has_cycle())

    print("\nPDF - dijkstra() example 1")
    print("--------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    for i in range(5):
        print(f'DIJKSTRA {i} {g.dijkstra(i)}')
    g.remove_edge(4, 3)
    print('\n', g)
    for i in range(5):
        print(f'DIJKSTRA {i} {g.dijkstra(i)}')
