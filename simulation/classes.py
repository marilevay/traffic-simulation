import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import random

class Car:
    """A car agent that moves along edges of a road network.

    Parameters
    ----------
    start: hashable
        Node identifier in the road network where the car starts
    destination: hashable
        Node identifier in the road network where the car is trying to go
    """

    def __init__(self, start, destination):
        self.current_location = start  
        self.destination = destination
        self.path = []
        self.parked = False
        
        # Track position and speed
        self.current_edge = None  # (u, v) tuple or None if at node
        self.position_on_edge = 0.0  # Distance traveled along current edge (meters)
        self.speed = 5.0  # Current speed in kph

class TrafficNetwork:
    """Traffic simulation on an OSMnx road network with density-based congestion.

    Parameters
    ----------
    graph: networkx.MultiDiGraph, optional
        Road network graph (as returned by OSMnx). If None, call
        load_road_network() before simulating
    cars: list[Car], optional
        Initial list of cars. If None, an empty list is used and cars
        can be created later via init_cars() or simulate_traffic()
    num_cars: int, default 100
        Default number of cars to initialize when using init_cars() or
        simulate_traffic() without explicitly passing num_cars
    """

    def __init__(self, graph=None, cars=None, num_cars: int = 100, spacing_per_car: float = 5.0):
        self.graph = graph
        self.cars = cars if cars is not None else []
        self.num_cars = num_cars
        # Approximate space one car occupies along the road (meters per car)
        self.spacing_per_car = spacing_per_car

    def load_road_network(self, address=None, dist=100, network_type="drive"):
        """Load a road network around a given address using OSMnx.

        Parameters
        ----------
        address: str, optional
            Geocodable address string ("Market St, San Francisco, CA").
        dist: int, default 100
            Search radius around the address, in meters.
        network_type: str, default "drive"
            Type of street network to download ("drive", "walk").

        Returns
        -------
        networkx.MultiDiGraph
            The loaded road network assigned to self.graph
        """
        self.graph = ox.graph_from_address(
            address,
            dist=dist,
            network_type=network_type,
        )
        return self.graph
    
    def edge_car_counts(self):
        """Count how many cars are currently on each directed edge.

        Returns
        -------
        dict
            Mapping (u, v, 0) edge keys to the integer number of cars
            whose current_edge is (u, v)
        """
        counts = {(u, v, 0): 0 for u, v, _ in self.graph.edges(keys=True)}
        for car in self.cars:
            if car.current_edge:
                u, v = car.current_edge
                counts[(u, v, 0)] += 1
        return counts

    def add_travel_time_attribute(self, default_speed=30):
        """Add a "travel_time" edge attribute based on length and speed.

        Parameters
        ----------
        default_speed: float, default 30
            Default speed (in km/h) to assume when no maxspeed attribute
            is available on an edge

        Returns
        -------
        networkx.MultiDiGraph
            The graph with a "travel_time" attribute added to each edge
        """
        for u, v, k, data in self.graph.edges(data=True, keys=True):
            speed = data.get("maxspeed", default_speed)
            if isinstance(speed, list):
                speed = speed[0]
            if isinstance(speed, str):
                speed = speed.split(" ")[0]
            data["travel_time"] = data["length"] / (float(speed) * 1000 / 60)
        return self.graph   
    
    def compute_edge_density(self, edge):
        """Calculate traffic density on a directed edge.

        Density is defined as the ratio

        density = (cars_on_edge) / (edge_capacity),

        where edge_capacity is proportional to the edge length and
        assumes roughly 5 units of distance per car. A value near 0.0
        indicates an empty edge, values around 1.0 indicate a road at
        capacity, and values greater than 1.0 indicate congestion.

        Parameters
        ----------
        edge : tuple
            Edge specified as (u, v) node pair.

        Returns
        -------
        float
            Traffic density on that edge.
        """
        u, v = edge
        edge_length_m = self.graph[u][v][0]['length']  # OSMnx stores length in meters
        edge_capacity = edge_length_m / self.spacing_per_car  # cars that can fit along this edge
        
        # Count cars on this edge
        cars_on_edge = sum(1 for car in self.cars if car.current_edge == edge)
        
        return cars_on_edge / edge_capacity if edge_capacity > 0 else 0
    
    def update_car_speed(self, car, max_speed=5.0):
        """Adjust a car's speed based on the density of its current edge

        Parameters
        ----------
        car: Car
            The car whose speed is being updated.
        max_speed: float, default 5.0
            Free-flow speed (units consistent with edge lengths, like kph)
        """
        if car.current_edge is None:
            car.speed = max_speed
            return
        
        density = self.compute_edge_density(car.current_edge)
        
        # Speed decreases with density
        if density < 0.2:
            car.speed = max_speed  # Free flow
        elif density < 0.8:
            car.speed = max_speed * (1 - 0.5*density)  # Gradual slowdown
        else:
            car.speed = max_speed * 0.2  # Heavy traffic, but still moving
    
    def move_cars(self):
        """Advance all cars by one simulation step.

        This performs a simultaneous update in two phases:

        1. Update speeds for all non-parked cars based on current densities.
        2. Move each non-parked car along its current edge or, if at a node,
           compute/continue a shortest path toward its destination.
        """
        
        # PHASE 1: Update speeds for all cars based on current density
        for car in self.cars:
            if car.parked:
                continue
            self.update_car_speed(car)
        
        # PHASE 2: Move all cars based on their speeds
        for car in self.cars:
            if car.parked:
                continue

            # Check if the car has reached its destination
            if car.current_location == car.destination:
                car.parked = True
                continue

            # Compute path if needed
            if not car.path:
                try:
                    car.path = nx.shortest_path(
                        self.graph,
                        car.current_location,
                        car.destination,
                        weight="travel_time",
                    )
                    # Remove current location from path
                    if car.path and car.path[0] == car.current_location:
                        car.path.pop(0)
                # If no possible shortest path was found, mark the car as parked 
                except nx.NetworkXNoPath:
                    car.parked = True
                    continue

            # If at a node, start moving to next node
            if car.current_edge is None:
                next_node = car.path[0]
                car.current_edge = (car.current_location, next_node)
                car.position_on_edge = 0.0
            
            # Move along current edge
            edge_length = self.graph[car.current_edge[0]][car.current_edge[1]][0]['length']
            car.position_on_edge += car.speed  # Move forward
            
            # Check if reached end of edge
            if car.position_on_edge >= edge_length:
                # Arrived at next node
                car.current_location = car.current_edge[1]
                car.path.pop(0)  # Remove this node from path
                car.current_edge = None
                car.position_on_edge = 0.0

    def plot_road_network(self):
        """Plot the raw road network using OSMnx defaults

        Returns
        -------
            The figure and axes created by ox.plot_graph
        """
        fig, ax = ox.plot_graph(self.graph, figsize=(15, 15))
        return fig, ax

    def init_cars(self, num_cars: int | None = None):
        """Initialize cars with random start/destination pairs.

        Parameters
        ----------
        num_cars : int, optional
            Number of cars to create. If ``None``, uses ``self.num_cars``.
        """
        if num_cars is None:
            num_cars = self.num_cars

        nodes = list(self.graph.nodes())
        self.cars = [Car(random.choice(nodes), random.choice(nodes)) for _ in range(num_cars)]

    def simulate_traffic(self, num_cars=100, num_steps=10):
        """Run a simple traffic simulation for a fixed number of steps.

        Parameters
        ----------
        num_cars: int, default 100
            Number of cars to initialize for this run
        num_steps: int, default 10
            Number of discrete time steps to simulate

        Returns
        -------
        list[Car]
            The list of cars after the simulation completes
        """

        # Create cars with random starts and destinations
        self.init_cars(num_cars=num_cars)

        for step in range(num_steps):
            self.move_cars()

            # Print stats
            print(f"\nStep {step + 1}:")
            parked = sum(1 for car in self.cars if car.parked)
            moving = len(self.cars) - parked
            print(f"  Moving: {moving}, Parked: {parked}")
            
            # Show average speed
            avg_speed = sum(car.speed for car in self.cars if not car.parked) / max(moving, 1)
            print(f"  Average speed: {avg_speed:.2f} kph")

        return self.cars

    def plot_network_with_osmnx(self):
        """Plot the network with edges colored and scaled by traffic density.

        Edge colors are determined by the density on each edge relative to
        the maximum density observed across the network. Line widths are
        still proportional to the number of cars on each edge. A colorbar is
        added to show the mapping from color intensity to density.

        Returns
        -------
            The figure and axes created by ox.plot_graph
        """
        edge_colors = []

        counts = self.edge_car_counts()

        # Collect densities for all edges
        densities = []
        for u, v, _ in self.graph.edges(keys=True):
            density = self.compute_edge_density((u, v))
            densities.append(density)

        max_density = max(densities) if densities else 1.0
        if max_density <= 0:
            max_density = 1.0

        # Build colors per edge (constant width everywhere)
        for (u, v, _), density in zip(self.graph.edges(keys=True), densities):
            count = counts.get((u, v, 0), 0)
            norm_density = density / max_density if max_density > 0 else 0.0
            edge_colors.append(plt.cm.plasma(norm_density))

        fig, ax = ox.plot_graph(
            self.graph,
            figsize=(15, 15),
            edge_color=edge_colors,
            edge_linewidth=2.0,
            show=False,
            close=False,
        )

        # Add a colorbar that reflects density values
        sm = mpl.cm.ScalarMappable(
            cmap=mpl.cm.plasma,
            norm=mpl.colors.Normalize(vmin=0, vmax=max_density),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Traffic density (cars per capacity)")

        return fig, ax

    def describe_edge(self, edge_key):
        """Return a human-readable description of a road segment.

        Parameters
        ----------
        edge_key: tuple
            Edge key of the form (u, v, k) from the underlying MultiDiGraph.

        Returns
        -------
        dict
            Dictionary with street metadata and endpoint coordinates
        """

        u, v, k = edge_key
        data = self.graph[u][v][k]

        u_data = self.graph.nodes[u]
        v_data = self.graph.nodes[v]

        u_lat, u_lon = u_data.get("y"), u_data.get("x")
        v_lat, v_lon = v_data.get("y"), v_data.get("x")

        return {
            "u": u,
            "v": v,
            "k": k,
            "street_name": data.get("name", "unknown street"),
            "highway": data.get("highway", "unknown"),
            "osmid": data.get("osmid", "unknown"),
            "u_coord": (u_lat, u_lon),
            "v_coord": (v_lat, v_lon),
        }

    def plot_network_with_networkx(self):
        """Plot the network using NetworkX with edges colored by density

        This provides an alternative visualization that uses networkx.draw
        (or equivalently, nx.draw) directly on the underlying graph, with
        edge colors reflecting the current density on each edge

        Returns
        -------
        The figure and axes created by matplotlib
        """

        G = self.graph

        # Use OSMnx node coordinates if available, otherwise fall back to a layout
        nodes_data = list(G.nodes(data=True))
        if nodes_data and "x" in nodes_data[0][1] and "y" in nodes_data[0][1]:
            pos = {n: (d["x"], d["y"]) for n, d in nodes_data}
        else:
            pos = nx.spring_layout(G, seed=0)

        # Collect edges and their densities
        edge_list = []
        densities = []
        for u, v, _ in G.edges(keys=True):
            edge_list.append((u, v))
            densities.append(self.compute_edge_density((u, v)))

        max_density = max(densities) if densities else 1.0
        if max_density <= 0:
            max_density = 1.0

        norm = mpl.colors.Normalize(vmin=0, vmax=max_density)
        cmap = plt.cm.plasma
        edge_colors = [cmap(norm(d)) for d in densities]

        fig, ax = plt.subplots(figsize=(10, 10))
        nx.draw(G, pos=pos, ax=ax, node_size=1, edgelist=edge_list, edge_color=edge_colors, width=1.0, with_labels=False)

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Traffic density (cars per capacity)")

        return fig, ax

