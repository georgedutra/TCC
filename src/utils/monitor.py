import time
import psutil
import threading
from typing import List, Dict
import matplotlib.pyplot as plt

class ResourceMonitor:
    """
    A class for monitoring and tracking system resource usage (CPU and memory) over time.
    This class provides functionality to sample CPU and memory usage at intervals,
    collect statistics, and visualize the usage patterns through plots.
    Attributes:
        cpu_samples (List[float]): List of CPU usage percentages collected during monitoring.
        memory_samples (List[float]): List of memory usage percentages collected during monitoring.
        timestamps (List[float]): List of elapsed times (in seconds) when each sample was taken.
        start_time (float or None): The timestamp when monitoring started, or None if not started.
        _monitoring (bool): Flag to control continuous monitoring loop.
    Example:
        ```
        monitor = ResourceMonitor()
        monitor.start()
        # Perform some operations
        monitor.sample()
        stats = monitor.get_stats()
        monitor.plot(save_path='usage_plot.png')
        ```
    """
    def __init__(self):
        """Initialize the ResourceMonitor with empty data lists and no start time."""
        self.cpu_samples: List[float] = []
        self.memory_samples: List[float] = []
        self.timestamps: List[float] = []
        self.start_time = None
        self._monitoring = False
        
    def start(self):
        """Start monitoring"""
        self.start_time = time.time()
        self.cpu_samples = []
        self.memory_samples = []
        self.timestamps = []
        
    def sample(self):
        """Take a sample of current CPU and memory usage"""
        if self.start_time is None:
            self.start()
            
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        elapsed = time.time() - self.start_time
        
        self.cpu_samples.append(cpu_percent)
        self.memory_samples.append(memory_percent)
        self.timestamps.append(elapsed)

    def monitor_continuous(self, interval: float = 1.0):
        """Monitor resources continuously at fixed intervals until stopped.
        NOT parallelized; blocks the current thread.

        Args:
            interval (float): Time in seconds between samples. Defaults to 1.0.
        """
        if self.start_time is None:
            self.start()
        self._monitoring = True
        while self._monitoring:
            self.sample()
            time.sleep(interval)

    def start_background_monitoring(self, interval: float = 1.0):
        """Start monitoring in a background thread.
        Parallelized; does not block the current thread.

        Args:
            interval (float, optional): Time in seconds between samples. Defaults to 1.0.

        Returns:
            threading.Thread: The thread object for the background monitoring.
        """
        if self.start_time is None:
            self.start()
            
        self._monitoring = True
        thread = threading.Thread(target=self.monitor_continuous, args=(interval,), daemon=True)
        thread.start()
        return thread
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self._monitoring = False
        
    def get_stats(self) -> Dict[str, float]:
        """Get statistics of the monitoring session"""
        if not self.cpu_samples:
            return {}
            
        return {
            'avg_cpu': sum(self.cpu_samples) / len(self.cpu_samples),
            'max_cpu': max(self.cpu_samples),
            'avg_memory': sum(self.memory_samples) / len(self.memory_samples),
            'max_memory': max(self.memory_samples),
            'duration': self.timestamps[-1] if self.timestamps else 0
        }
    
    def save_samples(self, filepath: str):
        """Save current samples to a file.
        
        Args:
            filepath (str): Path to the .csv file where samples will be saved.
        """
        if not self.cpu_samples:
            print("No data to save")
            return
            
        with open(filepath, 'w') as f:
            f.write("timestamp,cpu_percent,memory_percent\n")
            for timestamp, cpu, memory in zip(self.timestamps, self.cpu_samples, self.memory_samples):
                f.write(f"{timestamp},{cpu},{memory}\n")
        
        print(f"Samples saved to {filepath}")

    def plot(self, save_path: str = None):
        """Plot CPU and memory usage over time.

        Args:
            save_path (str, optional): Path to save the .png plot image. Defaults to None.
        """
        
        if not self.cpu_samples:
            print("No data to plot")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # CPU plot
        ax1.plot(self.timestamps, self.cpu_samples, 'b-', label='CPU Usage')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('CPU Usage (%)')
        ax1.set_title('CPU Usage Over Time')
        ax1.grid(True)
        ax1.legend()
        
        # Memory plot
        ax2.plot(self.timestamps, self.memory_samples, 'r-', label='Memory Usage')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Memory Usage (%)')
        ax2.set_title('Memory Usage Over Time')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()