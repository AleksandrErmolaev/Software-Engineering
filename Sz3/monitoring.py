class RecSysMonitor:
    def __init__(self):
        self.metrics_history = {
            'latency': [],
            'throughput': [],
            'accuracy': [],
            'diversity': []
        }

    def track_latency(self, latency_ms):
        if latency_ms > 200:
            self.send_alert(f"High latency: {latency_ms}ms")

    def track_drift(self, current_performance, baseline_performance):
        if abs(current_performance - baseline_performance) > 0.1:
            self.trigger_retraining()