import matplotlib.animation as animation
import matplotlib.pyplot as plt
import osmnx as ox
from tqdm import tqdm


def run_multiple_simulations(network_factory, num_runs: int = 10, num_steps: int = 50):
    """Run multiple traffic simulations and average edge densities over runs.

    Parameters
    ----------
    network_factory: callable
        Zero-argument function that returns a freshly initialized
        TrafficNetwork with graph and cars already set up.
    num_runs: int, default 10
        Number of independent simulation runs to perform.
    num_steps: int, default 50
        Number of timesteps per run.

    Returns
    -------
    dict
        Mapping (u, v, k) edge keys to a list of average densities over
        time, where each list has length num_steps.
    """
    if num_runs <= 0:
        raise ValueError("num_runs must be positive")

    # First network instance defines the edge set/order
    network = network_factory()
    edge_keys = list(network.graph.edges(keys=True))

    # sums[(u, v, k)] -> list of length num_steps, accumulating densities
    sums = {ek: [0.0] * num_steps for ek in edge_keys}

    # Loop over runs with progress bar
    for run_idx in tqdm(range(num_runs), desc="Simulation runs", unit="run"):
        if run_idx > 0:
            network = network_factory()

        # For each timestep, move cars and accumulate densities
        for t in range(num_steps):
            network.move_cars()

            for (u, v, k) in edge_keys:
                density = network.compute_edge_density((u, v))
                sums[(u, v, k)][t] += float(density)

    # Convert sums to averages over runs
    averages = {}
    for ek, values in sums.items():
        averages[ek] = [v / num_runs for v in values]

    return averages


def plot_congestion_time_series(avg_densities, top_k: int = 5, ax=None, network=None):
    """Plot time series of density for the most congested edges.

    Parameters
    ----------
    avg_densities: dict
        Output of run_multiple_simulations, mapping edge keys to lists of
        average densities over time
    top_k: int, default 5
        Number of edges with highest mean density to plot
    ax: matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, a new figure and axes are
        created
    network: TrafficNetwork, optional
        If provided, its describe_edge(edge_key) method is used to build
        human-readable labels (e.g., street names) for plotted edges.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the congestion time series plot.
    """
    if not avg_densities:
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        return ax

    # Compute mean density for each edge over time
    scores = []
    for edge_key, series in avg_densities.items():
        mean_density = sum(series) / len(series) if series else 0.0
        scores.append((edge_key, mean_density))

    scores.sort(key=lambda x: x[1], reverse=True)
    rankings = scores[:top_k]

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    for edge_key, _ in rankings:
        series = avg_densities[edge_key]
        timesteps = range(len(series))

        if network is not None and hasattr(network, "describe_edge"):
            info = network.describe_edge(edge_key)
            street = info.get("street_name", "unknown street")
            label = f"{street} (k={edge_key[2]})"
        else:
            label = f"edge {edge_key[0]}→{edge_key[1]} (k={edge_key[2]})"

        ax.plot(timesteps, series, label=label)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Average density (cars / capacity)")
    ax.set_title("Most congested roads: density over time")
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, linestyle=":", alpha=0.5)

    return ax


def animate_traffic(network_factory, num_steps: int = 50, interval_ms: int = 200, save_path: str | None = None):
    """Create an animation of traffic congestion evolving over time.

    Parameters
    ----------
    network_factory: callable
        Zero-argument function that returns a freshly initialized
        TrafficNetwork with graph and cars already set up
    num_steps: int, default 50
        Number of frames (timesteps) in the animation
    interval_ms: int, default 200
        Delay between frames in milliseconds
    save_path: str, optional
        If provided, the animation is saved to this file path (for
        example, an mp4 or gif). Otherwise it's not saved

    Returns
    -------
        The created animation object
    """
    network = network_factory()
    edge_keys = list(network.graph.edges(keys=True))

    fig, ax = plt.subplots(figsize=(10, 10))

    def update(frame_idx: int):
        network.move_cars()

        counts = network.edge_car_counts()
        max_count = max(counts.values()) if counts else 1

        edge_colors = []
        edge_widths = []
        for (u, v, k) in edge_keys:
            count = counts.get((u, v, k), 0)
            norm = count / max_count if max_count > 0 else 0.0
            edge_colors.append(plt.cm.Reds(norm))
            edge_widths.append(1 + count)

        ax.clear()
        ox.plot_graph(network.graph, ax=ax, node_size=0, edge_color=edge_colors, edge_linewidth=edge_widths, show=False, close=False)
        ax.set_title(f"Traffic congestion – step {frame_idx + 1}")
        return ax.collections

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=num_steps,
        interval=interval_ms,
        blit=False,
    )

    if save_path is not None:
        anim.save(save_path)

    return anim

