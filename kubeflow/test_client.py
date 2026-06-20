"""
Kubeflow Trainer: Local Training Example
This script demonstrates how to run single-node training using the Local Process Backend.
"""

import sys

from kubeflow.trainer import CustomTrainer, LocalProcessBackendConfig, TrainerClient

# ── Telemetry setup ───────────────────────────────────────────────────────────
# This block lives in the application (test_client.py), never inside the SDK.
# The SDK only calls get_tracer() — it never touches providers or exporters.
#
# To switch backends, change the exporter inside build_tracer_provider().
# Everything else (resource, processor, propagator) stays the same.


def build_tracer_provider(
    service_name: str = "kubeflow-trainer-example",
    backend: str = "console",  # "jaeger" | "otlp-collector" | "console"
    endpoint: str = "http://localhost:4317",
    headers: dict = None,  # needed for Honeycomb, Grafana Cloud, etc.
):
    """
    Build and return a TracerProvider configured for the chosen backend.

    Args:
        service_name:  Shows up as the service name in Jaeger / Tempo / etc.
        backend:       Which exporter to use.
                         "jaeger"         → OTLP gRPC direct to Jaeger
                         "otlp-collector" → OTLP gRPC to an OTel Collector
                         "console"        → prints spans to stdout (debugging)
        endpoint:      The OTLP endpoint. Ignored for "console".
        headers:       Extra HTTP headers (API keys for cloud backends).

    Returns:
        A configured TracerProvider instance.
    """
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    # Resource describes this process in every span it emits.
    # service.name is the most important attribute — it is the label in Jaeger.
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "0.1.0",
        }
    )

    provider = TracerProvider(resource=resource)

    # ── Choose exporter based on backend ─────────────────────────────────────
    if backend in ("jaeger", "otlp-collector"):
        # Both Jaeger (>=1.35) and any OTel Collector accept OTLP over gRPC.
        # Switching from Jaeger to a Collector is just an endpoint change.
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(
            endpoint=endpoint,
            headers=headers or {},
            # insecure=True is fine for local Jaeger without TLS.
            # Remove this line (or set insecure=False) for any HTTPS endpoint.
            insecure=True,
        )

    elif backend == "console":
        # Prints every span as JSON to stdout.
        # Useful when you do not have Jaeger running locally.
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        exporter = ConsoleSpanExporter()

    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Choose from: 'jaeger', 'otlp-collector', 'console'."
        )

    # BatchSpanProcessor buffers spans and sends them in batches.
    # Use SimpleSpanProcessor only for debugging — it blocks on every span.
    provider.add_span_processor(BatchSpanProcessor(exporter))

    # W3C Trace Context is the standard propagation format.
    # This is what puts the traceparent header on outbound HTTP calls.
    set_global_textmap(TraceContextTextMapPropagator())

    return provider


# ─────────────────────────────────────────────────────────────────────────────


def train_mnist():
    """Train a CNN model on MNIST dataset using PyTorch"""
    import torch
    from torch import nn, optim
    import torch.nn.functional as fa
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4 * 4 * 50, 500)
            self.fc2 = nn.Linear(500, 10)

        def forward(self, x):
            x = fa.relu(self.conv1(x))
            x = fa.max_pool2d(x, 2, 2)
            x = fa.relu(self.conv2(x))
            x = fa.max_pool2d(x, 2, 2)
            x = x.view(-1, 4 * 4 * 50)
            x = fa.relu(self.fc1(x))
            x = self.fc2(x)
            return fa.log_softmax(x, dim=1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = Net().to(device)
    model = torch.compile(model)

    dataset = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(1, 3):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = fa.nll_loss(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )

    torch.save(model.state_dict(), "mnist_cnn.pt")
    print("Training is finished")


def main():
    print("Make sure you have the Kubeflow SDK installed:")
    print("pip install kubeflow")
    print("-" * 50)

    # ── Build the provider and pass it to TrainerClient ──────────────────────
    #
    # Current setup: Jaeger running locally on the default OTLP gRPC port.
    #
    # To switch to an OTel Collector:
    #   provider = build_tracer_provider(
    #       backend="otlp-collector",
    #       endpoint="http://my-collector:4317",
    #   )
    #
    # To switch to Grafana Cloud (needs auth header):
    #   provider = build_tracer_provider(
    #       backend="otlp-collector",
    #       endpoint="https://tempo-prod.grafana.net:443",
    #       headers={"Authorization": "Basic <base64-token>"},
    #   )
    #
    # To just print spans to stdout without any backend:
    #   provider = build_tracer_provider(backend="console")
    #
    provider = build_tracer_provider(
        service_name="kubeflow-trainer-example",
        # backend="jaeger",
        endpoint="http://localhost:4317",
    )
    # ─────────────────────────────────────────────────────────────────────────

    backend_config = LocalProcessBackendConfig(cleanup_venv=True)

    # Provider is passed here — the SDK never installs it globally.
    client = TrainerClient(
        backend_config=backend_config,
        tracer_provider=provider,
    )

    torch_runtime = None
    for runtime in client.list_runtimes():
        print(f"Available runtime: {runtime}")
        if runtime.name == "torch-distributed":
            torch_runtime = runtime

    if torch_runtime is None:
        print("Error: Could not find torch-distributed runtime")
        sys.exit(1)

    job_name = client.train(
        trainer=CustomTrainer(
            func=train_mnist,
            packages_to_install=["pip-system-certs", "torch", "torchvision"],
        ),
        runtime=torch_runtime,
    )

    job = client.get_job(job_name)
    print(f"\nJob: {job.name}, Status: {job.status}")

    print("\nTraining logs:")
    print("-" * 50)
    for logline in client.get_job_logs(job_name, follow=True):
        print(logline, end="")

    print("\n" + "-" * 50)
    print("Deleting job...")
    client.delete_job(job_name)
    print("Job deleted successfully")

    # Flush any buffered spans before the process exits.
    # BatchSpanProcessor buffers spans — without this, spans emitted close
    # to process exit may not be sent before the process terminates.
    provider.force_flush()


if __name__ == "__main__":
    main()
