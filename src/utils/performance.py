"""
Performance monitoring utilities for CollabGPT.

This module provides utilities for tracking and measuring performance metrics
across different components of the application.
"""

import time
from typing import Dict, List, Any, Optional, Callable
from functools import wraps
from contextlib import contextmanager
import threading


class PerformanceMonitor:
    """
    Monitors performance metrics across the application.
    """
    
    def __init__(self):
        """Initialize the performance monitor."""
        self._metrics = {}
        self._metrics_lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, unit: str = "ms") -> None:
        """
        Record a performance metric.
        
        Args:
            name: The name of the metric
            value: The metric value
            unit: The unit of measurement (default: ms)
        """
        with self._metrics_lock:
            if name not in self._metrics:
                self._metrics[name] = {
                    'count': 0,
                    'total': 0.0,
                    'min': float('inf'),
                    'max': float('-inf'),
                    'unit': unit,
                    'last': 0.0
                }
                
            self._metrics[name]['count'] += 1
            self._metrics[name]['total'] += value
            self._metrics[name]['min'] = min(self._metrics[name]['min'], value)
            self._metrics[name]['max'] = max(self._metrics[name]['max'], value)
            self._metrics[name]['last'] = value
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all recorded performance metrics.
        
        Returns:
            Dictionary containing all metrics
        """
        with self._metrics_lock:
            result = {}
            for name, data in self._metrics.items():
                avg = data['total'] / data['count'] if data['count'] > 0 else 0
                result[name] = {
                    'count': data['count'],
                    'avg': avg,
                    'min': data['min'] if data['min'] != float('inf') else 0,
                    'max': data['max'] if data['max'] != float('-inf') else 0,
                    'unit': data['unit'],
                    'last': data['last']
                }
            return result
    
    def get_metric(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific metric.
        
        Args:
            name: The name of the metric
            
        Returns:
            Dictionary containing metric data or None if not found
        """
        with self._metrics_lock:
            if name not in self._metrics:
                return None
                
            data = self._metrics[name]
            avg = data['total'] / data['count'] if data['count'] > 0 else 0
            
            return {
                'count': data['count'],
                'avg': avg,
                'min': data['min'] if data['min'] != float('inf') else 0,
                'max': data['max'] if data['max'] != float('-inf') else 0,
                'unit': data['unit'],
                'last': data['last']
            }
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        with self._metrics_lock:
            self._metrics = {}
    
    @contextmanager
    def measure_latency(self, name: str):
        """
        Context manager for measuring execution time.
        
        Args:
            name: The name of the metric
        """
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000.0
            self.record_metric(name, duration_ms)
    
    def latency_decorator(self, name: str):
        """
        Decorator for measuring function execution time.
        
        Args:
            name: The name of the metric
            
        Returns:
            Decorated function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.measure_latency(name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator


# Create a singleton instance
_performance_monitor = PerformanceMonitor()

def get_performance_monitor() -> PerformanceMonitor:
    """
    Get the global performance monitor instance.
    
    Returns:
        PerformanceMonitor instance
    """
    return _performance_monitor

@contextmanager
def measure_latency(name: str, monitor: Optional[PerformanceMonitor] = None):
    """
    Context manager for measuring execution time.
    
    Args:
        name: The name of the metric
        monitor: Optional performance monitor instance (uses global if None)
    """
    monitor = monitor or _performance_monitor
    with monitor.measure_latency(name):
        yield

def latency_decorator(name: str, monitor: Optional[PerformanceMonitor] = None):
    """
    Decorator for measuring function execution time.
    
    Args:
        name: The name of the metric
        monitor: Optional performance monitor instance (uses global if None)
        
    Returns:
        Decorated function
    """
    monitor = monitor or _performance_monitor
    return monitor.latency_decorator(name)