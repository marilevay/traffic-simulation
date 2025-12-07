"""
Theoretical Analysis of Traffic Networks using Edge Betweenness Centrality

We do theoretical analysis of traffic flow patterns by measuring
edge betweenness centrality. This approach predicts which road segments will
be most popular based on shortest path calculations, independent of actual
traffic simulation.

Edge betweenness centrality measures how often a road segment lies on the
shortest paths between other nodes, making it ideal for predicting traffic
flow patterns on specific roads.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from simulation.classes import TrafficNetwork


class TheoreticalTrafficAnalysis:
    """Theoretical analysis of traffic patterns using centrality measures.

    """

    def __init__(self, network: TrafficNetwork):
       
        self.network = network
        self.graph = network.graph
        self.edge_betweenness = None


    def compute_edge_betweenness(self, weight='travel_time'):
        """Calculate betweenness centrality for all edges.

        Edge betweenness measures how often an edge lies on shortest paths
        between all pairs of nodes. Higher values indicate more critical
        edges for network connectivity.

        """
        print("Computing edge betweenness centrality...")
        #using inbuild function to calculate edge betweenness centrality
        self.edge_betweenness = nx.edge_betweenness_centrality(
            self.graph,
            weight=weight,
            normalized=True
        )
        return self.edge_betweenness


    def get_top_edges(self, k=10):
        """
        Get the k edges with highest betweenness centrality.
        """
        if self.edge_betweenness is None:
            self.compute_edge_betweenness()

        sorted_edges = sorted(
            self.edge_betweenness.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_edges[:k]

    def analyze_network_structure(self):
        """Provide summary statistics about the network structure.

      
        """
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph),
        }

        if self.edge_betweenness is not None:
            scores = list(self.edge_betweenness.values())
            stats['edge_centrality_mean'] = np.mean(scores)
            stats['edge_centrality_std'] = np.std(scores)
            stats['edge_centrality_max'] = np.max(scores)

        return stats


    def plot_network_centrality(
        self,
        top_k: int = 20,
        figsize: tuple[int, int] = (15, 15),
        focus_on_top_edges: bool = True,
        padding: float = 0.0005,
    ):
        """
        Visualize the network with edges colored by betweenness centrality.

    
        """
        if self.edge_betweenness is None:
            self.compute_edge_betweenness()

        # Get top edges (full (u, v, k) tuples)
        top_edges = self.get_top_edges(top_k)
        top_edge_set = {edge_tuple for edge_tuple, _ in top_edges}

        # Prepare edge data
        edge_tuples = list(self.graph.edges(keys=True))
        edgelist = [(u, v) for u, v, _ in edge_tuples]
        edge_colors = []
        edge_widths = []

        max_centrality = max(self.edge_betweenness.values())
        denom = max_centrality if max_centrality > 0 else 1.0

        for edge_tuple in edge_tuples:
            centrality = self.edge_betweenness.get(edge_tuple, 0.0)
            norm_centrality = centrality / denom

            if edge_tuple in top_edge_set:
                edge_colors.append('red')
                edge_widths.append(3 + centrality * 5)
            else:
                edge_colors.append(plt.cm.viridis(norm_centrality))
                edge_widths.append(0.5 + centrality * 3)

        # create visualization
        fig, ax = plt.subplots(figsize=figsize)

    
        nodes_data = list(self.graph.nodes(data=True))
        if nodes_data and "x" in nodes_data[0][1] and "y" in nodes_data[0][1]:
            pos = {n: (d["x"], d["y"]) for n, d in nodes_data}
        else:
            pos = nx.spring_layout(self.graph, seed=0)

        # draw the network
        nx.draw_networkx_nodes(self.graph, pos, ax=ax, node_size=10, alpha=0.5, node_color='lightblue')
        nx.draw_networkx_edges(
            self.graph,
            pos,
            ax=ax,
            edgelist=edgelist,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.8,
        )

        if focus_on_top_edges and top_edge_set:
            focus_nodes = set()
            for u, v, _ in top_edge_set:
                focus_nodes.add(u)
                focus_nodes.add(v)

            coords = [pos[n] for n in focus_nodes if n in pos]
            if coords:
                xs, ys = zip(*coords)
                x_span = max(xs) - min(xs)
                y_span = max(ys) - min(ys)
                x_pad = max(x_span * 0.1, padding)
                y_pad = max(y_span * 0.1, padding)
                ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
                ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)

        # colorbar
        sm = mpl.cm.ScalarMappable(
            cmap=mpl.cm.viridis,
            norm=mpl.colors.Normalize(vmin=0, vmax=max_centrality)
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Edge Betweenness Centrality')

        ax.set_title(f'Network Edge Centrality (Top {top_k} highlighted in red)')
        ax.axis('off')

        return fig, ax

    def print_analysis_summary(self):
        """Print a summary of the theoretical analysis results focused on edges."""
        if self.edge_betweenness is None:
            self.compute_edge_betweenness()

        stats = self.analyze_network_structure()
        top_edges = self.get_top_edges(5)

        print("=" * 60)
        print("THEORETICAL TRAFFIC ANALYSIS SUMMARY")
        print("=" * 60)

        print(f"\nNetwork Structure:")
        print(f"  Nodes: {stats['num_nodes']}")
        print(f"  Edges: {stats['num_edges']}")
        print(f"  Density: {stats['density']:.4f}")
        print(f"  Connected: {stats['is_connected']}")

        print(f"\nEdge Centrality Statistics:")
        print(f"  Mean: {stats['edge_centrality_mean']:.6f}")
        print(f"  Std:  {stats['edge_centrality_std']:.6f}")
        print(f"  Max:  {stats['edge_centrality_max']:.6f}")

        print(f"\nTop 5 Most Central Edges:")
        for i, (edge_tuple, score) in enumerate(top_edges):
            if hasattr(self.network, 'describe_edge') and len(edge_tuple) >= 3:
                try:
                    edge_info = self.network.describe_edge(edge_tuple)
                    street_name = edge_info.get('street_name', 'Unknown')
                    print(f"  {i+1}. {street_name} ({edge_tuple[0]}→{edge_tuple[1]}): {score:.6f}")
                except:
                    print(f"  {i+1}. Edge {edge_tuple}: {score:.6f}")
            else:
                print(f"  {i+1}. Edge {edge_tuple}: {score:.6f}")


def run_theoretical_analysis(address="Market St, San Francisco, CA", dist=800):
    """
    Run complete theoretical analysis on a traffic network.
    """
    print(f"Loading traffic network around {address} (radius: {dist}m)...")

    # Create and load network
    network = TrafficNetwork()
    network.load_road_network(address, dist=dist, network_type='drive')
    network.add_travel_time_attribute()

    # Run analysis
    analysis = TheoreticalTrafficAnalysis(network)
    analysis.compute_edge_betweenness()

    return analysis


if __name__ == "__main__":
    # Run theoretical analysis
    analysis = run_theoretical_analysis(dist=400)

    # Print summary
    analysis.print_analysis_summary()

    # Create visualizations
    print("\nGenerating visualizations...")

    # Edge centrality plot
    fig1, ax1 = analysis.plot_network_centrality(top_k=10)
    plt.show()