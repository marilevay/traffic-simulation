"""
Side-by-side visualization of theoretical (edge betweenness) vs empirical
traffic congestion over the full network extent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from simulation.classes import TrafficNetwork
from simulation.helpers import plot_congestion_time_series, run_multiple_simulations
from theoretical_analysis import TheoreticalTrafficAnalysis

# type alias for edge keys in MultiDiGraph: (source_node, target_node, edge_index)
EdgeKey = Tuple[int, int, int]


@dataclass
class VisualizationConfig:
    """Configuration parameters for dual visualization plots.
    
    Attributes:
        top_k: Number of top edges to highlight in time series plots
        figsize: Figure dimensions (width, height) in inches
        cmap: Colormap name for edge visualization
        highlight_color: Hex color code for highlighting specific edges
        base_linewidth: Minimum line width for edge rendering
        linewidth_scale: Scaling factor for edge line widths
    """
    top_k: int = 5
    figsize: Tuple[int, int] = (14, 8)
    cmap: str = "Reds"
    highlight_color: str = "#33C3F0"
    base_linewidth: float = 1.0
    linewidth_scale: float = 6.0


def _metric_from_avg_densities(avg_densities: Dict[EdgeKey, Iterable[float]]) -> Dict[EdgeKey, float]:
    """Compute time-averaged metric from density time series.
    
    Args
    -----
        avg_densities: Dictionary mapping edge keys to time series of density values
        
    Returns
    -------
        Dictionary mapping edge keys to their mean density values
    """
    return {edge_key: float(np.mean(series)) for edge_key, series in avg_densities.items()}


def _draw_edge_metric(
    ax,
    graph,
    metric: Dict[EdgeKey, float],
    title: str,
    colorbar_label: str,
    normalize: bool = False,
):
    """Draw a network graph with edges colored by a metric value.
    
    Args
    -----
        ax: Matplotlib axis to draw on
        graph: NetworkX graph to visualize
        metric: Dictionary mapping edge keys to metric values (e.g., density, betweenness)
        title: Title for the subplot
        colorbar_label: Label for the colorbar
        normalize: If True, normalize metric values to [0, 1] range
    """
    edge_keys = list(graph.edges(keys=True))
    nodes_data = list(graph.nodes(data=True))
    
    # Extract node positions from graph attributes or use spring layout
    if nodes_data and "x" in nodes_data[0][1] and "y" in nodes_data[0][1]:
        pos = {n: (d["x"], d["y"]) for n, d in nodes_data}
    else:
        pos = nx.spring_layout(graph, seed=0)

    # Build edge list and corresponding metric values
    edge_list = []
    densities = []
    for (u, v, k) in edge_keys:
        edge_list.append((u, v))
        densities.append(metric.get((u, v, k), 0.0))

    # Handle empty graph case
    if not densities:
        ax.set_axis_off()
        ax.set_title(title)
        return

    # Determine maximum value for normalization
    max_density = max(densities)
    if max_density <= 0:
        max_density = 1.0

    # Normalize to relative scale if requested
    if normalize:
        densities = [d / max_density for d in densities]
        colorbar_label += " (relative)"
        max_density = 1.0

    # Create color mapping using reversed plasma colormap
    norm = mpl.colors.Normalize(vmin=0, vmax=max_density)
    cmap = plt.cm.plasma_r
    edge_colors = [cmap(norm(d)) for d in densities]

    # Draw the network graph
    nx.draw(
        graph,
        pos=pos,
        ax=ax,
        node_size=1,
        edgelist=edge_list,
        edge_color=edge_colors,
        width=2.0,
        with_labels=False,
    )

    # Add colorbar to show metric scale
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.01)
    cbar.set_label(colorbar_label)

    ax.set_title(title)
    ax.set_axis_off()


def visualize_theoretical_vs_empirical(
    address: str = "Esmeralda, Buenos Aires, Argentina",
    dist: int = 800,
    num_cars: int = 500,
    num_steps: int = 200,
    num_runs: int = 20,
    config: VisualizationConfig | None = None,
):
    """Create side-by-side comparison of theoretical vs empirical traffic patterns.
    
    Generates a 2x2 grid of visualizations comparing:
    - Top row: Spatial network maps showing edge betweenness (left) vs actual congestion (right)
    - Bottom row: Time series plots of top edges for theoretical (left) vs empirical (right)
    
    Args
    ----
        address: Geographic location for road network (geocodable address string)
        dist: Distance in meters to extend network from address center point
        num_cars: Number of vehicles to simulate in the traffic network
        num_steps: Number of simulation time steps per run
        num_runs: Number of independent simulation runs to average over
        config: Optional visualization configuration (uses defaults if None)
        
    Returns
    -------
        Tuple containing:
            - Figure object with all subplots
            - Tuple of four axes (theory map, empirical map, theory time series, empirical time series)
    """
    cfg = config or VisualizationConfig()

    # Load base road network from OpenStreetMap
    base_network = TrafficNetwork(num_cars=num_cars)
    base_network.load_road_network(address, dist=dist, network_type="drive")
    base_network.add_travel_time_attribute()
    base_network.init_cars()

    # Compute theoretical edge betweenness centrality
    analysis = TheoreticalTrafficAnalysis(base_network)
    edge_betweenness = analysis.compute_edge_betweenness()

    # Factory function to create fresh network instances for each simulation run
    def factory() -> TrafficNetwork:
        net = TrafficNetwork(num_cars=num_cars)
        net.graph = base_network.graph.copy()
        net.init_cars()
        return net

    # Run multiple simulations and collect empirical traffic density data
    avg_densities, all_runs = run_multiple_simulations(
        network_factory=factory,
        num_runs=num_runs,
        num_steps=num_steps,
    )
    empirical_metric = _metric_from_avg_densities(avg_densities)

    # Create 2x2 subplot grid (maps on top, time series on bottom)
    fig = plt.figure(figsize=(cfg.figsize[0], cfg.figsize[1] * 1.4))
    ax_theory = fig.add_subplot(2, 2, 1)  # Top-left: theoretical map
    ax_empirical = fig.add_subplot(2, 2, 2)  # Top-right: empirical map
    ax_theory_ts = fig.add_subplot(2, 2, 3)  # Bottom-left: theoretical time series
    ax_empirical_ts = fig.add_subplot(2, 2, 4)  # Bottom-right: empirical time series

    # Draw theoretical edge betweenness map
    _draw_edge_metric(
        ax_theory,
        base_network.graph,
        edge_betweenness,
        title="Edge Betweenness (Theoretical)",
        colorbar_label="Edge betweenness centrality",
        normalize=True,
    )

    # Draw empirical congestion map
    _draw_edge_metric(
        ax_empirical,
        base_network.graph,
        empirical_metric,
        title="Average Congestion (Empirical)",
        colorbar_label="Time-averaged traffic density",
        normalize=True,
    )

    # Plot empirical density time series for top-k edges
    plot_congestion_time_series(
        avg_densities,
        top_k=cfg.top_k,
        ax=ax_empirical_ts,
        network=base_network,
    )
    ax_empirical_ts.set_title("Top empirical edges – density over time")

    # Create constant time series for theoretical values (betweenness doesn't change over time)
    theory_series: Dict[EdgeKey, List[float]] = {}
    constant_length = num_steps if num_steps > 0 else 1
    for edge_key, value in edge_betweenness.items():
        theory_series[edge_key] = [value] * constant_length

    # Plot theoretical importance time series for top-k edges
    plot_congestion_time_series(
        theory_series,
        top_k=cfg.top_k,
        ax=ax_theory_ts,
        network=base_network,
    )
    ax_theory_ts.set_ylabel("Relative importance (normalized)")
    ax_theory_ts.set_title("Top theoretical edges – relative importance")

    fig.tight_layout()

    return fig, (ax_theory, ax_empirical, ax_theory_ts, ax_empirical_ts)


if __name__ == "__main__":
    # Run visualization with default parameters and display the result
    visualize_theoretical_vs_empirical()
    plt.show()
