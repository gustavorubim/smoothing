import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_datareader.data as web
from datetime import datetime
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.preprocessing import StandardScaler

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
        
    def calculate_smoothing(self):
        """Apply all smoothing methods"""
        # Standard LOWESS with different spans
        self.processed_data['lowess_0.1'] = self.calculate_lowess(span=0.1)
        self.processed_data['lowess_0.3'] = self.calculate_lowess(span=0.3)
        
        # Adaptive LOWESS
        self.processed_data['adaptive_lowess'] = self.calculate_adaptive_lowess()
        
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
            line=dict(color='red', width=2)
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
                    dict(label="LOWESS Only",
                         method="update",
                         args=[{"visible": [True, True, True, True]}])
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

if __name__ == "__main__":
    main()
