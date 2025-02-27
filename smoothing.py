import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_datareader.data as web
from datetime import datetime
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks

class SmoothingPipeline:
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        
    def acquire_data(self):
        """Download and prepare VIX data"""
        try:
            start_date = datetime(2019, 1, 1)
            end_date = datetime(2024, 10, 31)
            
            vix_data = web.DataReader('VIXCLS', 'fred', start_date, end_date)
            
            if len(vix_data) == 0:
                raise ValueError("No VIX data retrieved")
                
            vix_data.columns = ['raw']
            self.raw_data = vix_data.resample('M').mean()
            
            # Initialize processed dataframe
            self.processed_data = pd.DataFrame(index=self.raw_data.index)
            self.processed_data['raw'] = self.raw_data['raw']
            
        except Exception as e:
            print(f"Error retrieving VIX data: {str(e)}")
            return None
            
    def calculate_lowess(self, span=0.3):
        """Calculate standard LOWESS smoothing"""
        endog = self.raw_data['raw'].values
        exog = np.arange(len(self.raw_data))
        
        smoothed = lowess(endog, exog, frac=span, it=3, delta=0.0, return_sorted=False)
        return smoothed
        
    def calculate_adaptive_lowess(self):
        """Calculate adaptive LOWESS with variable bandwidth"""
        endog = self.raw_data['raw'].values
        exog = np.arange(len(self.raw_data))
        
        # Calculate gradients to detect rapid changes
        gradients = np.abs(np.gradient(endog))
        grad_weights = gradients / np.max(gradients)
        
        # Use smaller window for faster response to changes
        window = max(int(len(self.raw_data) * 0.05), 3)
        rolling_std = pd.Series(endog).rolling(window=window, center=True).std()
        
        # Combine gradient and volatility information for bandwidth
        bandwidths = rolling_std / rolling_std.max()
        bandwidths = bandwidths.fillna(bandwidths.mean())
        
        # Adjust bandwidths to be smaller at peaks (high gradient areas)
        bandwidths = 0.05 + 0.25 * (1 - grad_weights) * bandwidths
        
        smoothed = np.zeros_like(endog)
        for i in range(len(self.raw_data)):
            local_smooth = lowess(endog, exog, frac=bandwidths[i], it=3, delta=0.0, return_sorted=False)
            smoothed[i] = local_smooth[i]
            
        return smoothed
    
    def calculate_peak_sensitive_lowess(self):
        """Calculate LOWESS that better preserves peaks"""
        endog = self.raw_data['raw'].values
        exog = np.arange(len(self.raw_data))
        
        # Detect peaks
        # Prominence parameter controls how significant a peak needs to be
        peaks, _ = find_peaks(endog, prominence=0.5*np.std(endog))
        
        # Create a mask of peak regions (the peak itself and surrounding points)
        peak_mask = np.zeros_like(endog, dtype=bool)
        for peak in peaks:
            # Mark peak and 1 point on each side as peak region
            lower_idx = max(0, peak-1)
            upper_idx = min(len(endog)-1, peak+1)
            peak_mask[lower_idx:upper_idx+1] = True
        
        # Standard smoothing as a base
        base_smoothed = lowess(endog, exog, frac=0.2, it=3, delta=0.0, return_sorted=False)
        
        # Apply more aggressive local smoothing around peaks
        peak_vicinity_smoothed = np.zeros_like(endog)
        
        # Calculate adaptive bandwidths
        bandwidths = np.ones_like(endog, dtype=float) * 0.2  # default bandwidth
        
        # Calculate gradients
        gradients = np.abs(np.gradient(endog))
        norm_gradients = gradients / np.max(gradients)
        
        # Set much smaller bandwidth at high gradient points
        high_gradient_points = norm_gradients > 0.3
        bandwidths[high_gradient_points] = 0.1
        
        # Even smaller bandwidth at peaks
        bandwidths[peak_mask] = 0.05
        
        # Apply smoothing with varying bandwidths
        smoothed = np.zeros_like(endog)
        for i in range(len(endog)):
            local_smooth = lowess(endog, exog, frac=bandwidths[i], it=2, delta=0.0, return_sorted=False)
            smoothed[i] = local_smooth[i]
            
        # Further adjustment: For peak points, blend original and smoothed value
        # This ensures peaks aren't suppressed too much
        peak_weighting = np.ones_like(endog) * 0.15  # default weighting of raw data
        peak_weighting[peak_mask] = 0.4  # higher weighting of raw data at peaks
        
        # Blend raw and smoothed data according to weights
        enhanced_smoothed = (1 - peak_weighting) * smoothed + peak_weighting * endog
            
        return enhanced_smoothed
    
    def calculate_asymmetric_lowess(self):
        """Calculate asymmetric LOWESS that's more responsive to upward movements"""
        endog = self.raw_data['raw'].values
        exog = np.arange(len(self.raw_data))
        
        # Calculate standard smoothed curve as base
        base_smoothed = lowess(endog, exog, frac=0.2, it=3, delta=0.0, return_sorted=False)
        
        # Calculate differences between raw and smoothed
        residuals = endog - base_smoothed
        
        # Apply asymmetric weighting: preserve positive residuals (where raw > smoothed) more
        asymmetric_adjustment = np.zeros_like(residuals)
        positive_residuals = residuals > 0
        
        # Keep more of positive residuals (peaks)
        asymmetric_adjustment[positive_residuals] = 0.5 * residuals[positive_residuals]
        
        # Add back weighted residuals to get asymmetric smoothed curve
        asymmetric_smoothed = base_smoothed + asymmetric_adjustment
        
        return asymmetric_smoothed
        
    def calculate_smoothing(self):
        """Apply all smoothing methods"""
        # Standard LOWESS with different spans
        self.processed_data['lowess_0.1'] = self.calculate_lowess(span=0.1)
        self.processed_data['lowess_0.3'] = self.calculate_lowess(span=0.3)
        
        # Adaptive LOWESS
        self.processed_data['adaptive_lowess'] = self.calculate_adaptive_lowess()
        
        # New peak-sensitive methods
        self.processed_data['peak_sensitive'] = self.calculate_peak_sensitive_lowess()
        self.processed_data['asymmetric'] = self.calculate_asymmetric_lowess()
        
    def generate_visualization(self):
        """Create interactive Plotly visualization"""
        fig = go.Figure()
        
        # Raw data
        fig.add_trace(go.Scatter(
            x=self.processed_data.index,
            y=self.processed_data['raw'],
            mode='markers+lines',
            name='Raw Data',
            line=dict(color='lightgray', width=1),
            marker=dict(size=4),
            showlegend=True
        ))
        
        # LOWESS curves
        fig.add_trace(go.Scatter(
            x=self.processed_data.index,
            y=self.processed_data['lowess_0.1'],
            mode='lines',
            name='LOWESS (span=0.1)',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.processed_data.index,
            y=self.processed_data['lowess_0.3'],
            mode='lines',
            name='LOWESS (span=0.3)',
            line=dict(color='green', width=2)
        ))
        
        # Adaptive LOWESS
        fig.add_trace(go.Scatter(
            x=self.processed_data.index,
            y=self.processed_data['adaptive_lowess'],
            mode='lines',
            name='Adaptive LOWESS',
            line=dict(color='orange', width=2)
        ))
        
        # New methods
        fig.add_trace(go.Scatter(
            x=self.processed_data.index,
            y=self.processed_data['peak_sensitive'],
            mode='lines',
            name='Peak-Sensitive LOWESS',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.processed_data.index,
            y=self.processed_data['asymmetric'],
            mode='lines',
            name='Asymmetric LOWESS',
            line=dict(color='purple', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title='VIX Index Smoothing Comparison',
            xaxis_title='Date',
            yaxis_title='VIX Value',
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05
            ),
            width=1200,
            height=800,
            margin=dict(r=150)
        )
        
        # Add range slider
        fig.update_xaxes(rangeslider_visible=True)
        
        # Add buttons for curve visibility
        updatemenus = [
            dict(
                type="buttons",
                direction="right",
                x=0.7,
                y=1.15,
                showactive=True,
                buttons=list([
                    dict(label="All Curves",
                         method="update",
                         args=[{"visible": [True] * len(fig.data)}]),
                    dict(label="Raw Only",
                         method="update",
                         args=[{"visible": [True] + [False] * (len(fig.data)-1)}]),
                    dict(label="New Methods Only",
                         method="update",
                         args=[{"visible": [True, False, False, False, True, True]}]),
                    dict(label="Original Methods Only",
                         method="update",
                         args=[{"visible": [True, True, True, True, False, False]}])
                ]),
            )
        ]
        
        fig.update_layout(updatemenus=updatemenus)
        
        return fig

def main():
    # Initialize and run pipeline
    pipeline = SmoothingPipeline()
    pipeline.acquire_data()
    
    if pipeline.raw_data is None:
        print("Failed to retrieve data")
        return
        
    pipeline.calculate_smoothing()
    fig = pipeline.generate_visualization()
    
    # Save to HTML for interactive viewing
    fig.write_html("vix_smoothing_analysis.html")
    
    # Show summary statistics
    print("\nSummary Statistics:")
    print(pipeline.processed_data.describe())
    
    # Compare peak preservation
    raw_max = pipeline.processed_data['raw'].max()
    methods = ['lowess_0.1', 'lowess_0.3', 'adaptive_lowess', 'peak_sensitive', 'asymmetric']
    
    print("\nPeak Preservation Analysis:")
    print(f"Raw data maximum: {raw_max:.2f}")
    
    for method in methods:
        method_max = pipeline.processed_data[method].max()
        preservation = method_max / raw_max * 100
        print(f"{method}: max={method_max:.2f}, preservation={preservation:.1f}%")

if __name__ == "__main__":
    main()
