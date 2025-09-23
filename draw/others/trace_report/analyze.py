import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from dataclasses import dataclass
import os
@dataclass
class TimeConfig:
    bin_seconds: int = 10

class TraceAnalyzer:
    def __init__(self, trace_file: str, time_config: TimeConfig):
        self.trace_file = trace_file
        self.time_config = time_config
        self.df = None
        self.start_time = None
        self.end_time = None
        self.seconds_at_target = None
        self.interval_markers = []
        self.save_dir = "draw/trace_report"

    def load_and_prepare_data(self) -> None:
        """Load and prepare the trace data for analysis."""
        self.df = pd.read_csv(self.trace_file)
        print(f"Data loaded with {len(self.df)} rows and columns: {self.df.columns.tolist()}")
        self.df['datetime'] = pd.to_datetime(self.df['TIMESTAMP'])
        self.df = self.df.sort_values(by='datetime')
        self.start_time = self.df['datetime'].min()
        self.end_time = self.df['datetime'].max()
        self.df['seconds_from_start'] = (self.df['datetime'] - self.start_time).dt.total_seconds()


    def format_time(self, seconds: float, _) -> str:
        """Format seconds to HH:MM time string."""
        time = self.start_time + timedelta(seconds=seconds)
        return time.strftime('%H:%M')

    def add_time_markers(self, start_time_str: str = "18:20", interval_minutes: int = 5) -> None:
        """Add vertical lines at specified time intervals and show request counts.
        
        Args:
            start_time_str: Start time in HH:MM format
            interval_minutes: Interval between markers in minutes
        """
        # Convert start time string to datetime
        start_hour, start_minute = map(int, start_time_str.split(':'))
        start_datetime = self.start_time.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
        
        # Calculate seconds from start for the first marker
        first_marker_seconds = (start_datetime - self.start_time).total_seconds()
        
        # Add markers every interval_minutes
        current_seconds = first_marker_seconds
        while current_seconds < (self.end_time - self.start_time).total_seconds():
            if current_seconds >= 0:  # Only add markers that are within the plot range
                # Add vertical line
                plt.axvline(x=current_seconds, color='gray', linestyle='--', alpha=0.5)
                
                # Calculate request count in this interval
                next_seconds = current_seconds + (interval_minutes * 60)
                mask = (self.df['seconds_from_start'] >= current_seconds) & (self.df['seconds_from_start'] < next_seconds)
                request_count = len(self.df[mask])
                
                # Calculate time range for the label
                start_time = self.start_time + timedelta(seconds=current_seconds)
                end_time = self.start_time + timedelta(seconds=next_seconds)
                
                # Add text label with time range and request count
                plt.text(current_seconds, plt.ylim()[1] * 0.95, 
                        f'{start_time.strftime("%H:%M")} - {end_time.strftime("%H:%M")}\n{request_count} requests', 
                        rotation=90, 
                        verticalalignment='top',
                        horizontalalignment='center',
                        fontsize=8)
                
            current_seconds += interval_minutes * 60

    def plot_arrival_pattern(self) -> None:
        """Plot request arrival pattern histogram."""
        plt.figure(figsize=(14, 6))
        bins = np.arange(0, (self.end_time - self.start_time).total_seconds() + self.time_config.bin_seconds, 
                        self.time_config.bin_seconds)
        histogram, bin_edges = np.histogram(self.df['seconds_from_start'], bins=bins)
        
        print(f"Debug: Data range - Start: {self.start_time}, End: {self.end_time}")
        print(f"Debug: Total seconds: {(self.end_time - self.start_time).total_seconds()}")
        print(f"Debug: First few seconds_from_start values: {self.df['seconds_from_start'].head()}")
        
        plt.bar(bin_edges[:-1], histogram, width=self.time_config.bin_seconds, alpha=0.7, align='edge')
        
        # Add time markers
        self.add_time_markers()
        
        plt.xlabel('Time (HH:MM)')
        plt.ylabel('Number of Requests')
        plt.title('Request Arrival Pattern (10s intervals)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'request_arrival_pattern.png'), dpi=300)
        plt.close()
        print("Saved request arrival pattern plot")

    def plot_cdf(self) -> None:
        """Plot CDF of request arrivals."""
        plt.figure(figsize=(14, 6))
        sorted_seconds = np.sort(self.df['seconds_from_start'])
        cdf = np.arange(1, len(sorted_seconds) + 1) / len(sorted_seconds)
        plt.plot(sorted_seconds, cdf, linewidth=2)
        plt.xlabel('Time (HH:MM)')
        plt.ylabel('Cumulative Probability')
        plt.title('CDF of Request Arrivals')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'request_arrival_cdf.png'), dpi=300)
        plt.close()
        print("Saved CDF plot")

    def analyze_time_range(self, start_seconds: float, end_seconds: float) -> None:
        """Analyze request counts in a specific time range.
        
        Args:
            start_seconds: Start time in seconds from the beginning
            end_seconds: End time in seconds from the beginning
        """
        mask = (self.df['seconds_from_start'] >= start_seconds) & (self.df['seconds_from_start'] < end_seconds)
        request_count = len(self.df[mask])
        
        start_time = self.start_time + timedelta(seconds=start_seconds)
        end_time = self.start_time + timedelta(seconds=end_seconds)
        
        print(f"\nTime Range Analysis:")
        print(f"From {start_time.strftime('%H:%M:%S')} to {end_time.strftime('%H:%M:%S')}")
        print(f"Total requests: {request_count}")
        print(f"Average requests per second: {request_count / (end_seconds - start_seconds):.2f}")

def main():
    time_config = TimeConfig(
        bin_seconds=10
    )
    
    analyzer = TraceAnalyzer(
        trace_file="/root/autodl-tmp/ada-place/datasets/AzureLLMInferenceTrace_code.csv",
        time_config=time_config
    )
    
    analyzer.load_and_prepare_data()
    analyzer.plot_arrival_pattern()
    analyzer.plot_cdf()
    
    # Analyze specific time range
    analyzer.analyze_time_range(500, 1500)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
