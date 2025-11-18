# Edgelens-CS
Edgelens-CS is a fog–cloud computing framework designed to enable real-time computer vision inference on resource-constrained IoT devices.
The system uses a distributed architecture, where lightweight Python clients capture data at the edge, while computationally expensive object-detection tasks are executed on the fog or cloud node.

The project implements:

- YOLO-based object detection (customized lightweight model)

- Edge offloading of inference for low-power devices

- Efficient communication layer between edge ↔ fog ↔ cloud

- Performance benchmarking for latency, throughput, and resource usage

- Modular components (app.py, yolo.py, benchmark.py) for deployment flexibility

This architecture significantly improves response time and energy efficiency compared to running ML locally on IoT devices. The framework is suitable for applications in smart surveillance, traffic monitoring, industrial IoT, and autonomous systems.
