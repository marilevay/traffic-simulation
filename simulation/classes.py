import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import random

class Car:
    """A simple car agent that moves along a precomputed shortest path.

    Attributes
    ----------
    current_location : hashable
        Current node in the road network.
    destination : hashable
        Target node where the car wants to park.
    path : list
        Remaining nodes in the current planned route.
    parked : bool
        Flag indicating whether the car has reached its destination or
        otherwise stopped moving.
    """
    def __init__(self, start, destination):
        self.current_location = start
        self.destination = destination
        self.path = []
        # FIX 1: Introducing a new attribute to mark if a car is parked
        self.parked = False


class TrafficNetwork:
    """Object-oriented wrapper around the underlying NetworkX road graph.

    Provides a convenient entry point to load the Berlin network, run the
    traffic simulation, and plot the resulting traffic patterns.
    """

    def __init__(self, graph=None, cars = []):
        self.graph = graph
        self.cars = cars

    def load_road_network(self, address=None, dist=1000, network_type="drive"):
        """
        ----------
        address : str, optional
            Address to download the road network from OSM.
        dist : int, optional
            Radius (in meters) around the address to download from OSM.
        network_type : str, optional
            OSMnx ``network_type`` argument (e.g. "drive").

        Returns
        -------
        networkx.MultiDiGraph
            Road network with a ``travel_time`` attribute on each edge.
        """
        # Get the road network for the specified location
        self.graph = ox.graph_from_address(
            address,
            dist=dist,
            network_type=network_type,
        )
        return self.graph
    
    def edge_car_counts(self):
        """Count how many cars intend to use each directed edge next step."""
        counts = {(u, v, 0): 0 for u, v, _ in self.graph.edges(keys=True)}
        for car in self.cars:
            if car.path:
                next_node = car.path[0]
                counts[(car.current_location, next_node, 0)] += 1
        return counts

    def add_travel_time_attribute(self, default_speed=30):
        """Attach a simple "travel_time" attribute to every edge in ``G``.

        The travel time is computed from edge length and a speed estimate. If
        ``maxspeed`` is missing, ``default_speed`` (km/h) is used.
        """
        # Add a 'travel_time' attribute to each edge to use as weight (for now, assume time is inversely proportional to speed limit, with a default of 30 km/h)
        for u, v, k, data in self.graph.edges(data=True, keys=True):
            speed = data.get("maxspeed", default_speed)
            if isinstance(speed, list):  #  There can be multiple speed limits; take the first one as representative
                speed = speed[0]
            if isinstance(speed, str):
                speed = speed.split(" ")[0]
            data["travel_time"] = data["length"] / (float(speed) * 1000 / 60)  # length in meters, speed in km/h, result in minutes
        return self.graph   
   
    
    def move_cars(self):
        """Advance all cars by one simulation step on graph ``G``.

        Parameters
        ----------
        G : networkx.Graph or DiGraph
            Road network whose edges must have a ``travel_time`` attribute.
        cars : list[Car]
            List of car agents whose state will be updated in-place.
        """
        for car in self.cars:
            # FIX 2: There may not always be a shortest path between two nodes. 
            # We assume we are taking the shortest path, in order to avoid the simulation becoming too computationally expensive 
            # When there is no shortest path between two nodes, we then assume this current car will park at the node it's in now 
            # We introduce a new try/except block to handle this case and make the simulation more robust

            # Check if the car is currently parked
            if car.parked:
                continue

            # Check if car has reached its destination
            if car.current_location == car.destination:
                car.parked = True
                continue

            if not car.path:
                try:
                    car.path = nx.shortest_path(
                        self.graph,
                        car.current_location,
                        car.destination,
                        weight="travel_time",
                    )
                except nx.NetworkXNoPath:
                    car.parked = True
                    continue

            next_node = car.path.pop(0)

            # Simulate traffic jam: if there are more than 'x' cars on an edge, slow down
            cars_on_edge = sum(
                1
                for c in self.cars
                if c.current_location == car.current_location
                and c.path
                and c.path[0] == next_node
            )
            if cars_on_edge > 5:  # Arbitrary threshold for a "jam"
                car.path.insert(0, next_node)  # Car stays in the same spot
            else:
                car.current_location = next_node

    def plot_network(self):
        """Plot the raw road network using OSMnx defaults."""
        fig, ax = ox.plot_graph(self.graph, figsize=(10, 10))
        return fig, ax

    def simulate_traffic(self, num_cars=100, num_steps=10):
        """
        Parameters
        ----------
        G : networkx.Graph or DiGraph
            Road network with ``travel_time`` edge attributes.
        num_cars : int, optional
            Number of cars to spawn with random origins and destinations.
        num_steps : int, optional
            Number of discrete time steps to simulate.

        Returns
        -------
        list[Car]
            Final list of cars with their end-state locations and paths.
        """

        # Create cars with random starts and destinations
        nodes = list(self.graph.nodes())
        self.cars = [Car(random.choice(nodes), random.choice(nodes)) for _ in range(num_cars)]

        for step in range(num_steps):
            self.move_cars()

            # For demonstration, print the number of cars at each node
            node_counts = {node: 0 for node in nodes}
            for car in self.cars:
                node_counts[car.current_location] += 1
            print(f"Step {step + 1}:")
            for node, count in node_counts.items():
                if count:
                    print(f"Node {node}: {count} cars")

        # FIX 3: We return the cars 
        return self.cars

    def plot_network_with_traffic(self):
        edge_colors = []
        edge_widths = []

        counts = self.edge_car_counts()
        max_count = max(counts.values())

        for u, v, _ in self.graph.edges(keys=True):
            count = counts.get((u, v, 0), 0)
            edge_colors.append(plt.cm.Reds(count / max_count))
            edge_widths.append(1 + count)

        fig, ax = ox.plot_graph(self.graph, figsize=(10,10), edge_color=edge_colors, edge_linewidth=edge_widths)
        return fig, ax



    