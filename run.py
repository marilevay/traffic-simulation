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
        net.load_road_network('Market St, San Francisco, CA', dist=dist, network_type='drive')
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

# Create traffic animation
def create_traffic_animation(
    num_steps: int = 50,
    num_cars: int = 5000,
    interval_ms: int = 200,
    save_path: str | None = None,
    dist: int = 200,
):
    """ Create a traffic animation with a specified number of steps, cars, and distance """
    def network_factory():
        net = TrafficNetwork(num_cars=num_cars)
        net.load_road_network('Market St, San Francisco, CA', dist=dist, network_type='drive')
        net.add_travel_time_attribute()
        net.init_cars()
        return net

    anim = animate_traffic(
        network_factory=network_factory,
        num_steps=num_steps,
        interval_ms=interval_ms,
        save_path=save_path,
    )
    plt.show()
    return anim

def plot_density_snapshot(
    num_steps: int = 20,
    num_cars: int = 1000,
    dist: int = 400,
    num_runs: int = 1,
    save_path: str | None = None,
):
    """Run simulation(s) and plot a static, time-averaged normalized density snapshot.

    This constructs TrafficNetwork instances via a small factory, runs one or
    more simulations, then computes the time-averaged density for each edge
    over all timesteps and runs. The final plot colors each edge according to
    this normalized, time-averaged density.
    """
    # Factory to build a fresh network for each run
    def network_factory():
        net = TrafficNetwork(num_cars=num_cars)
        net.load_road_network("Market St, San Francisco, CA", dist=dist, network_type="drive")
        net.add_travel_time_attribute()
        net.init_cars()
        return net

    # Average densities over time (and runs) using existing helper
    avg_densities = run_multiple_simulations(
        network_factory=network_factory,
        num_runs=num_runs,
        num_steps=num_steps,
    )

    # Time-average per edge: scalar density per edge
    time_avg = {}
    for edge_key, series in avg_densities.items():
        time_avg[edge_key] = sum(series) / len(series) if series else 0.0

    if not time_avg:
        # Fallback: nothing to plot
        net = network_factory()
        # show default OSMnx plot 
        fig, ax = net.plot_road_network()
        plt.show()
        return fig, ax

    # Normalize across edges
    max_density = max(time_avg.values())
    if max_density <= 0:
        max_density = 1.0

    # Build a sample network just for plotting (same graph structure)
    net = network_factory()
    edge_keys = list(net.graph.edges(keys=True))

    edge_colors = []
    edge_widths = []
    for ek in edge_keys:
        d = time_avg.get(ek, 0.0)
        norm = d / max_density
        edge_colors.append(plt.cm.Reds(norm))
        edge_widths.append(1.0 + 3.0 * norm)

    # Use NetworkX to plot with node coordinates if available
    G = net.graph
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

    # Normalize again for the colormap (uses same max_density)
    norm = mpl.colors.Normalize(vmin=0, vmax=max_density)
    cmap = plt.cm.Reds
    edge_colors = [cmap(norm(d)) for d in densities]

    fig, ax = plt.subplots(figsize=(15, 15))
    nx.draw(G, pos=pos, ax=ax, node_size=1, edgelist=edge_list, edge_color=edge_colors, width=2.0, with_labels=False)

    # Colorbar for time-averaged density
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Time-averaged traffic density (cars per capacity)")

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return fig, ax

if __name__ == "__main__":
    run_traffic_experiment(
        num_runs=5,
        num_steps=50,
        num_cars=1000,
        top_k=3,
        dist=400,
    )
    # NetworkX-based density snapshot
    plot_density_snapshot(
        num_steps=50,
        num_cars=1000,
        dist=400,
        num_runs=5,
        save_path="density_snapshot.png",
    )
    # Uncomment to also generate an animation video
    create_traffic_animation(
        num_steps=50,
        num_cars=1000,
        interval_ms=800,
        save_path="road_network.mp4",
        dist=400,
    )

