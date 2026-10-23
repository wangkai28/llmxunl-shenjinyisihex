from bioelectric.llm_provider import MockLLMProvider
from bioelectric.observer import Observer
from bioelectric.spiking import SpikingNetwork


def main():
    network = SpikingNetwork(n_units=64, seed=42)
    provider = MockLLMProvider(
        description="a network with a stable core and a quiet periphery",
        action_type="inject",
        region=0,
        strength=2.0,
    )
    observer = Observer(network, provider, observe_every=10, n_bins=8, mode="readwrite")
    reflections = observer.run(50)
    for i, reflection in enumerate(reflections):
        print(
            f"[{i}] {reflection.self_description} | "
            f"action={reflection.action_type} region={reflection.region} "
            f"strength={reflection.strength}"
        )


if __name__ == "__main__":
    main()
