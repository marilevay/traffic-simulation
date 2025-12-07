from simulation.classes import TrafficNetwork
from simulation.helpers import run_multiple_simulations, animate_traffic, plot_congestion_time_series
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx

def run_traffic_experiment(
    num_runs: int = 10,
    num_steps: int = 50,
    num_cars: int = 5000,
    top_k: int = 5,
    dist: int = 200,
):
    """ Run a traffic experiment with a specified number of runs, steps, cars, and distance """
    def network_factory():
        net = TrafficNetwork(num_cars=num_cars)
        net.load_road_network('Esmeralda, Buenos Aires, Argentina', dist=dist, network_type='drive')
        net.add_travel_time_attribute()
        net.init_cars()
        return net

    avg_densities = run_multiple_simulations(
        network_factory=network_factory,
        num_runs=num_runs,
        num_steps=num_steps,
    )
    # Use a sample network to provide street-level labels in the plot
    sample_network = network_factory()
    plot_congestion_time_series(avg_densities, top_k=top_k, network=sample_network)
    plt.show()
    return avg_densities

def create_traffic_animation(
    network,
    avg_densities,
    num_steps: int = 50,
    interval_ms: int = 200,
    save_path: str | None = None,
):
    """Create a traffic animation using pre-computed avg_densities.

    Parameters
    ----------
    network: TrafficNetwork
        A network instance providing the graph structure
    avg_densities: dict
        Mapping (u, v, k) -> list of density values over time
    num_steps: int, default 50
        Number of frames in the animation
    interval_ms: int, default 200
        Delay between frames in milliseconds
    save_path: str, optional
        If provided, save the animation to this path

    Returns
    -------
        The animation object
    """
    anim = animate_traffic(
        network=network,
        avg_densities=avg_densities,
        num_steps=num_steps,
        interval_ms=interval_ms,
        save_path=save_path,
    )
    plt.show()
    return anim

def plot_density_snapshot(
    network,
    avg_densities,
    save_path: str | None = None,
):
    """Plot a static, time-averaged normalized density snapshot.

    This function takes pre-computed avg_densities from run_multiple_simulations
    and visualizes the time-averaged density for each edge.

    Parameters
    ----------
    network: TrafficNetwork
        A network instance providing the graph structure
    avg_densities: dict
        Mapping (u, v, k) -> list of density values over time
    save_path: str, optional
        If provided, save the figure to this path

    Returns
    -------
    fig, ax
        The matplotlib figure and axes
    """
    # Time-average per edge: scalar density per edge
    time_avg = {}
    for edge_key, series in avg_densities.items():
        time_avg[edge_key] = sum(series) / len(series) if series else 0.0

    if not time_avg:
        # Fallback: nothing to plot
        fig, ax = network.plot_road_network()
        plt.show()
        return fig, ax

    # Normalize across edges
    max_density = max(time_avg.values())
    if max_density <= 0:
        max_density = 1.0

    # Use NetworkX to plot with node coordinates if available
    G = network.graph
    edge_keys = list(G.edges(keys=True))
    nodes_data = list(G.nodes(data=True))
    if nodes_data and "x" in nodes_data[0][1] and "y" in nodes_data[0][1]:
        pos = {n: (d["x"], d["y"]) for n, d in nodes_data}
    else:
        pos = nx.spring_layout(G, seed=0)

    # Edge list and densities in same order as edge_keys
    edge_list = []
    densities = []
    for (u, v, k) in edge_keys:
        edge_list.append((u, v))
        densities.append(time_avg.get((u, v, k), 0.0))

    # Normalize for the colormap (uses same max_density)
    # Use reversed plasma so the darkest colors correspond to highest density
    norm = mpl.colors.Normalize(vmin=0, vmax=max_density)
    cmap = plt.cm.plasma_r
    edge_colors = [cmap(norm(d)) for d in densities]

    fig, ax = plt.subplots(figsize=(15, 15))
    nx.draw(G, pos=pos, ax=ax, node_size=1, edgelist=edge_list, edge_color=edge_colors, width=2.0, with_labels=False)

    # Colorbar for time-averaged density (same plasma_r colormap)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Time-averaged traffic density (cars per capacity)")

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return fig, ax

if __name__ == "__main__":
    # Run the main experiment once to get avg_densities
    num_runs = 20
    num_steps = 800
    num_cars = 1000
    top_k = 3
    dist = 400

    # Factory to build a fresh network for each run
    def network_factory():
        net = TrafficNetwork(num_cars=num_cars)
        net.load_road_network('Esmeralda, Buenos Aires, Argentina', dist=dist, network_type='drive')
        net.add_travel_time_attribute()
        net.init_cars()
        return net

    # Run multiple simulations to get averaged densities
    avg_densities = run_multiple_simulations(
        network_factory=network_factory,
        num_runs=num_runs,
        num_steps=num_steps,
    )

    # Use a sample network for plotting and labeling
    sample_network = network_factory()

    # STEP 1: Time-series congestion plot
    plot_congestion_time_series(avg_densities, top_k=top_k, network=sample_network, save_path="congestion_time_series.png")
    plt.show()

    # STEP 2: NetworkX-based density snapshot
    plot_density_snapshot(
        network=sample_network,
        avg_densities=avg_densities,
        save_path="density_snapshot.png",
    )

    # STEP 3: Traffic animation video
    create_traffic_animation(
        network=sample_network,
        avg_densities=avg_densities,
        num_steps=num_steps,
        interval_ms=800,
        save_path="road_network.mp4",
    )

