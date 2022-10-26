# Course: CS261 - Data Structures
# Author: Sebastian Gajardo
# Assignment: Assignment 6 Graphs
# Description: Un-directed Graph implementation using an adjacency list ADT as the basis with add vertex, add edge,
# remove edge, remove vertex, get vertices, get edges, is valid path, depth first search, breadth first search, count
# connected components and has cycle methods.

import heapq
from collections import deque


class UndirectedGraph:
    """
    Class to implement undirected graph
    - duplicate edges not allowed
    - loops not allowed
    - no edge weights
    - vertex names are strings
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency list
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.adj_list = dict()

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            for u, v in start_edges:
                self.add_edge(u, v)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        out = [f'{v}: {self.adj_list[v]}' for v in self.adj_list]
        out = '\n  '.join(out)
        if len(out) < 70:
            out = out.replace('\n  ', ', ')
            return f'GRAPH: {{{out}}}'
        return f'GRAPH: {{\n  {out}}}'

    # ------------------------------------------------------------------ #

    def add_vertex(self, v: str) -> None:
        """
        Receives vertex name and adds this new vertex to the graph. If a vertex with the same name already exist in the
        graph it does nothing.
        """
        if v not in self.adj_list:  # If vertex not found, add vertex to graph.
            self.adj_list[v] = []
        
    def add_edge(self, u: str, v: str) -> None:
        """
        Receives two vertex names and adds an edge connecting the two vertices. If either or both vertex names do not
        exists in the graph it creates them and then creates the edge. If the edge already exists in the graph or if u
        and v refer to the same vertex it does nothing.
        """
        if v != u:  # If u and v not same vertex.
            if u not in self.adj_list:  # If u not in graph, add to graph.
                self.adj_list[u] = []
            if v not in self.adj_list:  # If v not in graph add to graph.
                self.adj_list[v] = []

            if v not in self.adj_list[u]:  # If edge does not exist, add edge.
                self.adj_list[u].append(v)
                self.adj_list[v].append(u)

    def remove_edge(self, v: str, u: str) -> None:
        """
        Receives two vertex names and removes edge connecting them. If either or both vertexes do not exist in the graph
        or there's no edge between them the method does nothing.
        """
        if (v in self.adj_list and u in self.adj_list) and u in self.adj_list[v]:  # If both vertexes in graph and edge
            self.adj_list[v].remove(u)  # exist, remove edge.
            self.adj_list[u].remove(v)

    def remove_vertex(self, v: str) -> None:
        """
        Receives a vertex name and removes that vertex from the graph and all edges incident to it from the graph. If
        the vertex does not exist the method does nothing.
        """
        if v in self.adj_list:  # If vertex in graph, remove it.
            v_list = self.adj_list[v]  # Save all edges, prior to removal.
            self.adj_list.pop(v)

            for vertex in v_list:  # For each vertex, remove edge that was connected to removed vertex.
                self.adj_list[vertex].remove(v)

    def get_vertices(self) -> []:
        """
        Return list of vertices in the graph (any order).
        """
        vertices = []
        for vertex in self.adj_list:  # For each vertex add it to vertices list.
            vertices.append(vertex)

        return vertices  # Return all vertices in the graph.

    def get_edges(self) -> []:
        """
        Return list of edges in the graph (any order). Each edge is returned as tuple of two incident vertex names
        """
        edge_list = []
        for vertex in self.adj_list:  # For each vertex check all edges.
            for connection in self.adj_list[vertex]:

                if (vertex, connection) and (connection, vertex) not in edge_list:  # If edge not in list, add to list.
                    edge_list.append((vertex, connection))

        return edge_list  # Return list of tuples representing edges in graph.

    def is_valid_path(self, path: []) -> bool:
        """
        Receives a list of vertex names and returns True if sequence of vertices represents a valid path, an empty path
        is considered valid. With a valid path you can travel from the first vertex to the last vertex at each step
        traversing over an edge.
        """
        if len(path) == 1 and path[0] not in self.adj_list:  # If path contains one edge and not in graph return False.
            return False

        valid = True  # Boolean if path is valid.
        n = 0  # Counter.

        while valid is True and n < (len(path) - 1):  # While valid True and n < (len - 1), check if vertex in graph and
            if path[n] not in self.adj_list or path[n + 1] not in self.adj_list[path[n]]:  # If connected to next vertex
                valid = False  # if not change valid to False and end loop.

            n += 1  # Increase counter.

        return valid  # Return if valid path boolean.

    def dfs(self, v_start, v_end=None) -> []:
        """
        Receives the vertex at which to start and an optional parameter of the vertex at which to end (default of None).
        Then performs a depth first search and returns the list of vertices in the order they are visited, if starting
        vertex not in graph returns empty list and if name of end vertex provided but not in graph search is done as if
        there is no end vertex.
        """
        visited = []  # List of visited vertexes.
        stack = deque(v_start)  # Deque to be used as a stack.

        if v_start not in self.adj_list:  # If starting vertex not in graph return empty list.
            return visited

        while len(stack) != 0:  # While stack not empty, pop top of stack.
            vertex = stack.pop()
            if vertex not in visited:  # If first time encountering vertex add to visited.
                visited.append(vertex)
                if vertex == v_end:  # If appended vertex is target vertex return visited.
                    return visited

                reverse_order = sorted(self.adj_list[vertex], reverse=True)  # Reverse sort neighbors of vertex.
                for neighbor in reverse_order:  # Append neighbors to stack in reverse sorted order.
                    stack.append(neighbor)

        return visited  # Return vertices in visited order.

    def bfs(self, v_start, v_end=None) -> []:
        """
        Receives the vertex at which to start and an optional parameter of the vertex at which to end (default of None).
        Then performs a breadth first search and returns the list of vertices in the order they are visited, if starting
        vertex not in graph returns empty list and if name of end vertex provided but not in graph search is done as if
        there is no end vertex.
        """
        visited = []  # List of visited vertexes.
        queue = deque(v_start)  # Deque to be used as a queue.

        if v_start not in self.adj_list:  # If starting vertex not in graph return empty list.
            return visited

        while len(queue) != 0:  # While queue not empty, dequeue first item.
            vertex = queue.popleft()
            if vertex not in visited:  # If first time encountering vertex add to visited and add neighbors to queue.
                visited.append(vertex)
                if vertex == v_end:  # If appended vertex is target vertex return visited.
                    return visited

                sorted_order = sorted(self.adj_list[vertex])  # Sort neighbors of vertex.
                for neighbor in sorted_order:  # Enqueue neighbors to queue in sorted order.
                    queue.append(neighbor)

        return visited  # Return vertices in visited order.

    def count_connected_components(self):
        """
        Returns number of connected components in the graph.
        """
        c_c = 0  # Connected component counter.
        visited = []  # Visited holder.

        for vertex in self.adj_list:  # For each vertex in graph, if vertex not in visited add one to cc counter.
            if vertex not in visited:
                c_c += 1
            visited += self.bfs(vertex)  # Add returned list of visited vertices to visited.

        return c_c  # Return number of connected components.

    def has_cycle(self):
        """
        Return True if there's at least one cycle in the graph, False otherwise.
        """
        unvisited = self.get_vertices()  # List of unvisited vertices.
        visited = []  # List of visited vertexes.
        stack = deque()  # Deque to be used as stack.

        while len(unvisited) != 0:  # While all vertices not checked, add first of unvisited to stack.
            stack.append((None, unvisited[0]))

            while len(stack) != 0:  # While stack not empty pop top of stack in parent, vertex format.
                parent, vertex = stack.pop()

                if vertex not in visited:  # If first time encountering vertex add to visited, remove from unvisited.
                    visited.append(vertex)
                    unvisited.remove(vertex)

                    reverse_order = sorted(self.adj_list[vertex], reverse=True)  # Reverse sort neighbors of vertex.
                    for neighbor in reverse_order:  # Append neighbors to stack in reverse sorted order.

                        if neighbor != parent:  # If neighbor vertex not parent, then add to stack.
                            stack.append((vertex, neighbor))

                elif vertex != parent:  # If vertex visited and not equal to parent, cycle found.
                    return True

        return False  # Return False if no cycle found in graph.


if __name__ == '__main__':

    print("\nPDF - method add_vertex() / add_edge example 1")
    print("----------------------------------------------")
    g = UndirectedGraph()
    print(g)

    for v in 'ABCDE':
        g.add_vertex(v)
    print(g)

    g.add_vertex('A')
    print(g)

    for u, v in ['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE', ('B', 'C'), "DD"]:
        g.add_edge(u, v)
    print(g)


    print("\nPDF - method remove_edge() / remove_vertex example 1")
    print("----------------------------------------------------")
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    g.remove_vertex('DOES NOT EXIST')
    g.remove_edge('A', 'B')
    g.remove_edge('X', 'B')
    print(g)
    g.remove_vertex('D')
    print(g)


    print("\nPDF - method get_vertices() / get_edges() example 1")
    print("---------------------------------------------------")
    g = UndirectedGraph()
    print(g.get_edges(), g.get_vertices(), sep='\n')
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE'])
    print(g.get_edges(), g.get_vertices(), sep='\n')


    print("\nPDF - method is_valid_path() example 1")
    print("--------------------------------------")
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    test_cases = ['ABC', 'ADE', 'ECABDCBE', 'ACDECB', '', 'D', 'Z']
    for path in test_cases:
        print(list(path), g.is_valid_path(list(path)))


    print("\nPDF - method dfs() and bfs() example 1")
    print("--------------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = 'ABCDEGH'
    for case in test_cases:
        print(f'{case} DFS:{g.dfs(case)} BFS:{g.bfs(case)}')
    print('-----')
    for i in range(1, len(test_cases)):
        v1, v2 = test_cases[i], test_cases[-1 - i]
        print(f'{v1}-{v2} DFS:{g.dfs(v1, v2)} BFS:{g.bfs(v1, v2)}')


    print("\nPDF - method count_connected_components() example 1")
    print("---------------------------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print(g.count_connected_components(), end=' ')
    print()


    print("\nPDF - method has_cycle() example 1")
    print("----------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG',
        'add FG', 'remove GE')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print('{:<10}'.format(case), g.has_cycle())
