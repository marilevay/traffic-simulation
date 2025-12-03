from simulation.classes import TrafficNetwork
from simulation.helpers import run_multiple_simulations, animate_traffic, plot_congestion_time_series
import matplotlib.pyplot as plt

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
):
    """Run a single simulation and plot a static density snapshot.

    This constructs one TrafficNetwork, runs a short simulation, and then
    plots the final network state using the current plot_network method"""

    net = TrafficNetwork(num_cars=num_cars)
    net.load_road_network("Market St, San Francisco, CA", dist=dist, network_type="drive")
    net.add_travel_time_attribute()
    net.init_cars()
    net.simulate_traffic(num_cars=num_cars, num_steps=num_steps)

    fig, ax = net.plot_network()
    plt.show()
    return fig, ax


if __name__ == "__main__":
    run_traffic_experiment(
        num_runs=5,
        num_steps=50,
        num_cars=1000,
        top_k=3,
        dist=800,
    )
    # NetworkX-based density snapshot
    plot_density_snapshot(
        num_steps=50,
        num_cars=1000,
        dist=800,
    )
    # Uncomment to also generate an animation video
    create_traffic_animation(
        num_steps=50,
        num_cars=1000,
        interval_ms=800,
        save_path="road_network.mp4",
        dist=800,
    )
