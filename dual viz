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
from simulation.helpers import run_multiple_simulations
from theoretical_analysis import TheoreticalTrafficAnalysis

EdgeKey = Tuple[int, int, int]


@dataclass
class VisualizationConfig:
    top_k: int = 10
    figsize: Tuple[int, int] = (14, 7)
    cmap: str = "Reds"
    highlight_color: str = "#33C3F0"
    base_linewidth: float = 1.0
    linewidth_scale: float = 6.0


def _edge_positions(graph) -> Tuple[List[List[Tuple[float, float]]], Dict[int, Tuple[float, float]]]:
    nodes = list(graph.nodes(data=True))
    if not nodes:
        raise ValueError("Graph must contain nodes with coordinates.")

    if "x" in nodes[0][1] and "y" in nodes[0][1]:
        pos = {n: (d["x"], d["y"]) for n, d in nodes}
    else:
        raise ValueError("Graph nodes lack geographic coordinates (x/y).")

    lines: List[List[Tuple[float, float]]] = []
    for u, v, _ in graph.edges(keys=True):
        if u in pos and v in pos:
            lines.append([pos[u], pos[v]])
    return lines, pos


def _metric_from_avg_densities(avg_densities: Dict[EdgeKey, Iterable[float]]) -> Dict[EdgeKey, float]:
    return {edge_key: float(np.mean(series)) for edge_key, series in avg_densities.items()}


def _draw_edge_metric(
    ax,
    graph,
    metric: Dict[EdgeKey, float],
    title: str,
    colorbar_label: str,
    normalize: bool = False,
):
    edge_keys = list(graph.edges(keys=True))
    nodes_data = list(graph.nodes(data=True))
    if nodes_data and "x" in nodes_data[0][1] and "y" in nodes_data[0][1]:
        pos = {n: (d["x"], d["y"]) for n, d in nodes_data}
    else:
        pos = nx.spring_layout(graph, seed=0)

    edge_list = []
    densities = []
    for (u, v, k) in edge_keys:
        edge_list.append((u, v))
        densities.append(metric.get((u, v, k), 0.0))

    if not densities:
        ax.set_axis_off()
        ax.set_title(title)
        return

    max_density = max(densities)
    if max_density <= 0:
        max_density = 1.0

    if normalize:
        densities = [d / max_density for d in densities]
        colorbar_label += " (relative)"
        max_density = 1.0

    norm = mpl.colors.Normalize(vmin=0, vmax=max_density)
    cmap = plt.cm.plasma_r
    edge_colors = [cmap(norm(d)) for d in densities]

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

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.01)
    cbar.set_label(colorbar_label)

    ax.set_title(title)
    ax.set_axis_off()


def visualize_theoretical_vs_empirical(
    address: str = "Turk Street, San Francisco, CA",
    dist: int = 400,
    num_cars: int = 100,
    num_steps: int = 80,
    num_runs: int = 20,
    config: VisualizationConfig | None = None,
):
    cfg = config or VisualizationConfig()

    base_network = TrafficNetwork(num_cars=num_cars)
    base_network.load_road_network(address, dist=dist, network_type="drive")
    base_network.add_travel_time_attribute()
    base_network.init_cars()

    analysis = TheoreticalTrafficAnalysis(base_network)
    edge_betweenness = analysis.compute_edge_betweenness()

    def factory() -> TrafficNetwork:
        net = TrafficNetwork(num_cars=num_cars)
        net.graph = base_network.graph.copy()
        net.init_cars()
        return net

    avg_densities = run_multiple_simulations(
        network_factory=factory,
        num_runs=num_runs,
        num_steps=num_steps,
    )
    empirical_metric = _metric_from_avg_densities(avg_densities)

    fig, (ax_theory, ax_empirical) = plt.subplots(1, 2, figsize=cfg.figsize, constrained_layout=True)

    _draw_edge_metric(
        ax_theory,
        base_network.graph,
        edge_betweenness,
        title="Edge Betweenness (Theoretical)",
        colorbar_label="Edge betweenness centrality",
        normalize=True,
    )

    _draw_edge_metric(
        ax_empirical,
        base_network.graph,
        empirical_metric,
        title="Average Congestion (Empirical)",
        colorbar_label="Time-averaged traffic density",
        normalize=True,
    )

    return fig, (ax_theory, ax_empirical)


if __name__ == "__main__":
    visualize_theoretical_vs_empirical()
    plt.show()
