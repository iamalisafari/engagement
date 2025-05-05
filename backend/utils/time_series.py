"""
Time-Series Analysis for Engagement Patterns

This module implements time series analysis techniques for engagement data
based on academic research methodologies. It provides functions for pattern
recognition, forecasting, and comparative analysis of temporal engagement data.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger("utils.time_series")


def identify_temporal_pattern(
    time_series: pd.DataFrame,
    timestamp_col: str = "timestamp",
    value_col: str = "value"
) -> Dict[str, Any]:
    """
    Identify temporal patterns in engagement data.
    
    This applies pattern recognition techniques based on academic research
    to identify common engagement patterns (e.g., early peak, sustaining, etc.).
    
    Args:
        time_series: DataFrame containing time series data
        timestamp_col: Column name for timestamps
        value_col: Column name for engagement values
        
    Returns:
        Dictionary of identified pattern characteristics
    """
    # Ensure data is sorted by timestamp
    time_series = time_series.sort_values(by=timestamp_col)
    
    # Extract values as numpy array
    values = time_series[value_col].values
    
    # Minimum viable analysis requires at least 5 data points
    if len(values) < 5:
        return {
            "pattern_type": "insufficient_data",
            "confidence": 0.0
        }
    
    # Normalize values to 0-1 range for better comparison
    min_val = np.min(values)
    max_val = np.max(values)
    
    # Handle flat line case
    if max_val == min_val:
        return {
            "pattern_type": "flat",
            "confidence": 1.0,
            "trend": "stable",
            "volatility": 0.0
        }
    
    # Normalize values
    norm_values = (values - min_val) / (max_val - min_val)
    
    # Calculate basic statistics
    mean_val = np.mean(norm_values)
    std_val = np.std(norm_values)
    skewness = stats.skew(norm_values)
    kurtosis = stats.kurtosis(norm_values)
    
    # Calculate trend using linear regression
    x = np.arange(len(norm_values))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, norm_values)
    
    # Calculate volatility (normalized standard deviation)
    volatility = std_val / mean_val if mean_val > 0 else 0
    
    # Determine trend direction
    if abs(slope) < 0.01:
        trend = "stable"
    elif slope > 0:
        trend = "increasing"
    else:
        trend = "decreasing"
    
    # Find peaks and troughs
    peaks = []
    troughs = []
    
    for i in range(1, len(norm_values) - 1):
        if norm_values[i] > norm_values[i-1] and norm_values[i] > norm_values[i+1]:
            # Found a peak
            peaks.append((i, norm_values[i]))
        elif norm_values[i] < norm_values[i-1] and norm_values[i] < norm_values[i+1]:
            # Found a trough
            troughs.append((i, norm_values[i]))
    
    # Sort peaks by height
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    # Define pattern types based on characteristics
    pattern_type = "undefined"
    confidence = 0.5  # Default confidence
    
    # Calculate position of global max and min
    max_pos = np.argmax(norm_values) / len(norm_values)
    min_pos = np.argmin(norm_values) / len(norm_values)
    
    # Pattern identification logic
    if volatility < 0.1 and abs(slope) < 0.01:
        pattern_type = "sustained"
        confidence = 0.8
    elif max_pos < 0.3 and trend == "decreasing":
        pattern_type = "early_peak"
        confidence = 0.7 + (norm_values[0] - norm_values[-1]) * 0.3
    elif max_pos > 0.7 and trend == "increasing":
        pattern_type = "late_peak"
        confidence = 0.7 + (norm_values[-1] - norm_values[0]) * 0.3
    elif min_pos > 0.3 and min_pos < 0.7 and norm_values[0] > mean_val and norm_values[-1] > mean_val:
        pattern_type = "u_shaped"
        confidence = 0.6 + min(norm_values[0], norm_values[-1]) * 0.4
    elif max_pos > 0.3 and max_pos < 0.7 and norm_values[0] < mean_val and norm_values[-1] < mean_val:
        pattern_type = "inverted_u"
        confidence = 0.6 + (1 - max(norm_values[0], norm_values[-1])) * 0.4
    elif len(peaks) >= 3:
        pattern_type = "multi_peak"
        # Higher confidence if peaks are more significant
        peak_significance = sum(p[1] for p in peaks[:3]) / 3
        confidence = 0.5 + peak_significance * 0.5
    elif volatility > 0.3:
        pattern_type = "volatile"
        confidence = 0.5 + min(0.5, volatility - 0.3)
    elif r_value**2 > 0.7:  # Strong linear trend
        pattern_type = f"linear_{trend}"
        confidence = 0.5 + r_value**2 * 0.5
    
    # Build result dictionary
    result = {
        "pattern_type": pattern_type,
        "confidence": float(confidence),
        "trend": trend,
        "slope": float(slope),
        "r_squared": float(r_value**2),
        "volatility": float(volatility),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "peak_positions": [float(p[0]/len(norm_values)) for p in peaks[:3]] if peaks else []
    }
    
    # Add more detailed statistics
    if len(peaks) > 0:
        result["highest_peak_position"] = float(peaks[0][0] / len(norm_values))
        result["highest_peak_value"] = float(peaks[0][1])
    
    # Calculate stationarity test
    try:
        adf_result = adfuller(norm_values)
        result["stationary"] = adf_result[1] < 0.05  # p-value less than 0.05 indicates stationarity
        result["adf_pvalue"] = float(adf_result[1])
    except:
        result["stationary"] = None
        result["adf_pvalue"] = None
    
    return result


def detect_seasonality(
    time_series: pd.DataFrame,
    timestamp_col: str = "timestamp",
    value_col: str = "value",
    freq: Optional[str] = None
) -> Dict[str, Any]:
    """
    Detect seasonality patterns in time series data.
    
    Args:
        time_series: DataFrame containing time series data
        timestamp_col: Column name for timestamps
        value_col: Column name for engagement values
        freq: Frequency string for decomposition (e.g., 'D', 'H')
        
    Returns:
        Dictionary with seasonality information
    """
    # Ensure timestamp is datetime and sorted
    time_series = time_series.copy()
    if time_series[timestamp_col].dtype != 'datetime64[ns]':
        time_series[timestamp_col] = pd.to_datetime(time_series[timestamp_col])
    
    time_series = time_series.sort_values(by=timestamp_col)
    
    # Get values
    values = time_series[value_col].values
    
    # Need a reasonable number of data points
    if len(values) < 10:
        return {
            "has_seasonality": False,
            "confidence": 0.0,
            "reason": "insufficient_data"
        }
    
    # Set time series index
    ts = time_series.set_index(timestamp_col)[value_col]
    
    # Auto-detect frequency if not provided
    if freq is None:
        # Try to infer frequency from time differences
        diff = ts.index.to_series().diff().median()
        hours = diff.total_seconds() / 3600
        
        if hours < 1:
            freq = 'H'  # Hourly
        elif hours < 24:
            freq = 'D'  # Daily
        elif hours < 24*7:
            freq = 'W'  # Weekly
        else:
            freq = 'M'  # Monthly
    
    # Ensure the time series has a regular frequency
    ts = ts.asfreq(freq, method='ffill')
    
    try:
        # Check for strong seasonality using autocorrelation
        acf_values = acf(values, nlags=min(len(values)-1, 50))
        
        # Find peaks in ACF
        acf_peaks = []
        for i in range(2, len(acf_values)-1):
            if acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1]:
                acf_peaks.append((i, acf_values[i]))
        
        # Sort peaks by correlation value
        acf_peaks.sort(key=lambda x: x[1], reverse=True)
        
        # Check for seasonal patterns
        seasonality_detected = False
        seasonal_period = None
        seasonality_strength = 0.0
        
        if acf_peaks and acf_peaks[0][1] > 0.3:  # Significant autocorrelation
            seasonal_period = acf_peaks[0][0]
            seasonality_strength = acf_peaks[0][1]
            seasonality_detected = True
        
        # Try to decompose series for additional confirmation
        decomposition_result = None
        if len(ts) >= max(2*seasonal_period if seasonal_period else 0, 10):
            try:
                # Use detected period or a sensible default
                period = seasonal_period if seasonal_period else min(len(ts)//2, 7)
                
                # Perform decomposition
                decomposition = seasonal_decompose(ts, model='additive', period=period)
                
                # Measure strength of seasonality
                seasonal_strength = 1.0 - np.var(decomposition.resid) / np.var(decomposition.seasonal + decomposition.resid)
                seasonal_strength = max(0, min(1, seasonal_strength))
                
                # Update overall strength estimate
                if seasonal_strength > seasonality_strength:
                    seasonality_strength = seasonal_strength
                    seasonality_detected = True
                
                decomposition_result = {
                    "seasonal_strength": float(seasonal_strength),
                    "trend_strength": float(1.0 - np.var(decomposition.resid) / np.var(decomposition.trend + decomposition.resid)),
                    "residual_variance": float(np.var(decomposition.resid) / np.var(ts))
                }
            except Exception as e:
                logger.warning(f"Error in seasonal decomposition: {e}")
        
        return {
            "has_seasonality": seasonality_detected,
            "confidence": float(seasonality_strength),
            "period": seasonal_period,
            "peaks": [(int(p[0]), float(p[1])) for p in acf_peaks[:3]] if acf_peaks else [],
            "decomposition": decomposition_result
        }
        
    except Exception as e:
        logger.error(f"Error detecting seasonality: {e}")
        return {
            "has_seasonality": False,
            "confidence": 0.0,
            "error": str(e)
        }


def forecast_engagement(
    time_series: pd.DataFrame,
    forecast_horizon: int = 10,
    timestamp_col: str = "timestamp",
    value_col: str = "value",
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Forecast future engagement values using time series models.
    
    Args:
        time_series: DataFrame containing time series data
        forecast_horizon: Number of periods to forecast
        timestamp_col: Column name for timestamps
        value_col: Column name for engagement values
        confidence_level: Confidence level for prediction intervals
        
    Returns:
        Dictionary with forecast values and confidence intervals
    """
    # Ensure timestamp is datetime and sorted
    time_series = time_series.copy()
    if time_series[timestamp_col].dtype != 'datetime64[ns]':
        time_series[timestamp_col] = pd.to_datetime(time_series[timestamp_col])
    
    time_series = time_series.sort_values(by=timestamp_col)
    
    # Create a DataFrame with the value column and datetime index
    ts_df = time_series.set_index(timestamp_col)[[value_col]]
    
    # Need sufficient data for forecasting
    if len(ts_df) < 5:
        return {
            "success": False,
            "error": "Insufficient data for forecasting (need at least 5 data points)",
            "forecast_values": [],
            "confidence_intervals": []
        }
    
    try:
        # Detect seasonality
        seasonality_result = detect_seasonality(time_series, timestamp_col, value_col)
        
        # Determine ARIMA parameters (simplified approach)
        p, d, q = 1, 1, 0  # Default ARIMA parameters
        seasonal_order = None
        
        # Check for non-stationarity
        adf_result = adfuller(ts_df[value_col].values)
        is_stationary = adf_result[1] < 0.05
        
        if is_stationary:
            d = 0
        
        # Check for seasonality
        if seasonality_result["has_seasonality"] and seasonality_result["confidence"] > 0.5:
            seasonal_period = seasonality_result["period"]
            if seasonal_period:
                seasonal_order = (1, 0, 0, seasonal_period)
        
        # Train ARIMA model
        if seasonal_order:
            # Use SARIMA
            model = sm.tsa.statespace.SARIMAX(
                ts_df[value_col], 
                order=(p, d, q),
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
        else:
            # Use ARIMA
            model = ARIMA(ts_df[value_col], order=(p, d, q))
        
        model_fit = model.fit()
        
        # Generate forecast
        forecast_result = model_fit.get_forecast(steps=forecast_horizon)
        forecast_values = forecast_result.predicted_mean
        
        # Generate prediction intervals
        alpha = 1 - confidence_level
        forecast_ci = forecast_result.conf_int(alpha=alpha)
        
        # Generate future timestamps
        last_date = ts_df.index[-1]
        # Try to infer frequency
        date_diff = ts_df.index[-1] - ts_df.index[-2]
        
        future_dates = [last_date + (i+1)*date_diff for i in range(forecast_horizon)]
        
        # Format results
        forecast_series = pd.Series(forecast_values, index=future_dates)
        ci_df = pd.DataFrame(
            forecast_ci, 
            index=future_dates,
            columns=['lower', 'upper']
        )
        
        return {
            "success": True,
            "model_type": "SARIMA" if seasonal_order else "ARIMA",
            "model_params": {
                "p": p, "d": d, "q": q,
                "seasonal_order": seasonal_order
            },
            "forecast_values": [
                {"timestamp": date.strftime("%Y-%m-%d %H:%M:%S"), "value": float(val)}
                for date, val in forecast_series.items()
            ],
            "confidence_intervals": [
                {
                    "timestamp": date.strftime("%Y-%m-%d %H:%M:%S"),
                    "lower": float(row["lower"]),
                    "upper": float(row["upper"])
                }
                for date, row in ci_df.iterrows()
            ],
            "confidence_level": confidence_level,
            "AIC": float(model_fit.aic) if hasattr(model_fit, 'aic') else None
        }
        
    except Exception as e:
        logger.error(f"Error in forecasting: {e}")
        return {
            "success": False,
            "error": str(e),
            "forecast_values": [],
            "confidence_intervals": []
        }


def compare_temporal_patterns(
    time_series_list: List[pd.DataFrame],
    identifiers: List[str],
    timestamp_col: str = "timestamp",
    value_col: str = "value",
    normalize: bool = True
) -> Dict[str, Any]:
    """
    Compare temporal engagement patterns across multiple content items.
    
    Args:
        time_series_list: List of DataFrames containing time series data
        identifiers: List of content identifiers matching time_series_list
        timestamp_col: Column name for timestamps
        value_col: Column name for engagement values
        normalize: Whether to normalize values for comparison
        
    Returns:
        Dictionary with comparison results
    """
    if not time_series_list or len(time_series_list) != len(identifiers):
        return {
            "success": False,
            "error": "Invalid input: need matching time series and identifiers"
        }
    
    # Analyze each time series
    pattern_results = []
    normalized_series = []
    
    for i, ts_df in enumerate(time_series_list):
        ts_id = identifiers[i]
        
        # Ensure timestamp is datetime and sorted
        ts_df = ts_df.copy()
        if ts_df[timestamp_col].dtype != 'datetime64[ns]':
            ts_df[timestamp_col] = pd.to_datetime(ts_df[timestamp_col])
        
        ts_df = ts_df.sort_values(by=timestamp_col)
        
        # Skip if not enough data
        if len(ts_df) < 5:
            pattern_results.append({
                "id": ts_id,
                "pattern": "insufficient_data",
                "similarity_scores": {}
            })
            continue
        
        # Store values
        values = ts_df[value_col].values
        
        # Normalize if requested
        if normalize:
            min_val = np.min(values)
            max_val = np.max(values)
            if max_val > min_val:
                normalized = (values - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(values)
        else:
            normalized = values
        
        # Store normalized series
        normalized_series.append((ts_id, normalized))
        
        # Analyze pattern
        pattern = identify_temporal_pattern(ts_df, timestamp_col, value_col)
        pattern_results.append({
            "id": ts_id,
            "pattern": pattern["pattern_type"],
            "pattern_details": pattern,
            "similarity_scores": {}
        })
    
    # Calculate pairwise similarities
    for i, (id_i, norm_i) in enumerate(normalized_series):
        for j, (id_j, norm_j) in enumerate(normalized_series):
            if i != j:
                # Calculate similarity using correlation
                # Interpolate to same length if needed
                if len(norm_i) != len(norm_j):
                    # Create interpolation indices
                    xi = np.linspace(0, 1, len(norm_i))
                    xj = np.linspace(0, 1, len(norm_j))
                    
                    # Interpolate the shorter series to match the longer one
                    if len(norm_i) > len(norm_j):
                        norm_j_interp = np.interp(xi, xj, norm_j)
                        norm_i_compare = norm_i
                        norm_j_compare = norm_j_interp
                    else:
                        norm_i_interp = np.interp(xj, xi, norm_i)
                        norm_i_compare = norm_i_interp
                        norm_j_compare = norm_j
                else:
                    norm_i_compare = norm_i
                    norm_j_compare = norm_j
                
                # Calculate correlation
                corr, p_value = stats.pearsonr(norm_i_compare, norm_j_compare)
                
                # Calculate DTW distance for shape similarity
                dtw_distance = 0.0  # Simplified - would use actual DTW in full implementation
                
                # Store similarity score
                pattern_results[i]["similarity_scores"][id_j] = {
                    "correlation": float(corr),
                    "p_value": float(p_value),
                    "shape_distance": float(dtw_distance)
                }
    
    # Find clusters of similar patterns
    clusters = find_pattern_clusters(pattern_results)
    
    return {
        "success": True,
        "patterns": pattern_results,
        "clusters": clusters
    }


def find_pattern_clusters(pattern_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Find clusters of similar engagement patterns.
    
    This is a simplified implementation that uses correlation scores
    to group patterns into clusters.
    
    Args:
        pattern_results: List of pattern analysis results
        
    Returns:
        List of cluster information
    """
    # Need at least 3 patterns to form meaningful clusters
    if len(pattern_results) < 3:
        return [{
            "cluster_id": "all",
            "members": [p["id"] for p in pattern_results],
            "pattern_type": "insufficient_data" if len(pattern_results) == 0 else pattern_results[0]["pattern"],
            "confidence": 0.0
        }]
    
    # Extract correlation matrix for clustering
    ids = [p["id"] for p in pattern_results]
    patterns = [p["pattern"] for p in pattern_results]
    
    # Simple clustering by pattern type
    pattern_clusters = {}
    for i, pattern in enumerate(patterns):
        if pattern not in pattern_clusters:
            pattern_clusters[pattern] = []
        pattern_clusters[pattern].append(ids[i])
    
    # Convert to output format
    clusters = []
    for i, (pattern, members) in enumerate(pattern_clusters.items()):
        clusters.append({
            "cluster_id": f"cluster_{i+1}",
            "pattern_type": pattern,
            "members": members,
            "size": len(members),
            "confidence": 1.0 if len(members) > 1 else 0.5
        })
    
    # Sort clusters by size
    clusters.sort(key=lambda x: x["size"], reverse=True)
    
    return clusters


def detect_viral_potential(
    time_series: pd.DataFrame,
    timestamp_col: str = "timestamp",
    value_col: str = "value",
    window_size: int = 5
) -> Dict[str, Any]:
    """
    Detect viral potential in the early stages of engagement.
    
    This analyzes growth rate, acceleration, and pattern characteristics
    to estimate viral potential.
    
    Args:
        time_series: DataFrame containing time series data
        timestamp_col: Column name for timestamps
        value_col: Column name for engagement values
        window_size: Window size for calculating growth metrics
        
    Returns:
        Dictionary with viral potential indicators
    """
    # Ensure timestamp is datetime and sorted
    time_series = time_series.copy()
    if time_series[timestamp_col].dtype != 'datetime64[ns]':
        time_series[timestamp_col] = pd.to_datetime(time_series[timestamp_col])
    
    time_series = time_series.sort_values(by=timestamp_col)
    
    # Need enough data points
    if len(time_series) < window_size + 2:
        return {
            "success": False,
            "error": f"Insufficient data (need at least {window_size + 2} data points)",
            "viral_score": 0.0
        }
    
    # Get values and calculate differences
    values = time_series[value_col].values
    
    # Calculate rolling metrics
    viral_indicators = {}
    
    # 1. Growth rate (average percentage change)
    pct_changes = []
    for i in range(1, len(values)):
        if values[i-1] > 0:
            pct_changes.append((values[i] - values[i-1]) / values[i-1])
        else:
            pct_changes.append(0)
    
    # Average growth rate over rolling windows
    rolling_growth_rates = []
    for i in range(len(pct_changes) - window_size + 1):
        window = pct_changes[i:i+window_size]
        rolling_growth_rates.append(np.mean(window))
    
    # 2. Growth acceleration (second derivative)
    growth_acceleration = np.diff(rolling_growth_rates)
    
    # 3. Convexity (how viral patterns tend to be convex early on)
    # Calculate polynomial fit
    x = np.arange(len(values))
    poly_coeffs = np.polyfit(x, values, 2)
    
    # Extract convexity from quadratic coefficient
    convexity = poly_coeffs[0]
    
    # Compile metrics
    viral_indicators["avg_growth_rate"] = float(np.mean(pct_changes))
    viral_indicators["latest_growth_rate"] = float(pct_changes[-1]) if pct_changes else 0.0
    viral_indicators["peak_growth_rate"] = float(np.max(pct_changes)) if pct_changes else 0.0
    
    viral_indicators["growth_acceleration"] = float(np.mean(growth_acceleration)) if len(growth_acceleration) > 0 else 0.0
    viral_indicators["convexity"] = float(convexity)
    
    # Calculate viral score (simplified model)
    # Combine metrics with appropriate weighting
    viral_score = (
        0.4 * min(1.0, viral_indicators["avg_growth_rate"] * 5) +
        0.3 * min(1.0, viral_indicators["latest_growth_rate"] * 5) +
        0.2 * min(1.0, viral_indicators["growth_acceleration"] * 10) +
        0.1 * min(1.0, viral_indicators["convexity"] * 100)
    )
    
    # Ensure score is in [0, 1] range
    viral_score = max(0.0, min(1.0, viral_score))
    
    # Determine viral category
    if viral_score > 0.8:
        viral_category = "highly_viral"
    elif viral_score > 0.6:
        viral_category = "viral"
    elif viral_score > 0.4:
        viral_category = "potentially_viral"
    elif viral_score > 0.2:
        viral_category = "moderate_growth"
    else:
        viral_category = "stable"
    
    return {
        "success": True,
        "viral_score": float(viral_score),
        "viral_category": viral_category,
        "metrics": viral_indicators
    } 