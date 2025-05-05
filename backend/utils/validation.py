"""
Statistical Validation Methods

This module implements statistical validation techniques to ensure
the reliability and validity of engagement analysis results, following
established scientific methodologies for research validation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

# Configure logging
logger = logging.getLogger("utils.validation")


def calculate_confidence_intervals(
    data: np.ndarray,
    confidence_level: float = 0.95,
    method: str = "bootstrap",
    num_bootstraps: int = 1000
) -> Dict[str, Any]:
    """
    Calculate confidence intervals for engagement metrics.
    
    Args:
        data: Array of engagement values
        confidence_level: Desired confidence level (0-1)
        method: Method to use ('bootstrap', 'parametric', or 't')
        num_bootstraps: Number of bootstrap samples if using bootstrap method
        
    Returns:
        Dictionary with confidence interval information
    """
    # Input validation
    if len(data) < 2:
        return {
            "success": False,
            "error": "Insufficient data for confidence interval calculation",
            "lower": None,
            "upper": None
        }
    
    try:
        if method == "bootstrap":
            # Bootstrap method (non-parametric)
            bootstrap_means = []
            for _ in range(num_bootstraps):
                sample = np.random.choice(data, size=len(data), replace=True)
                bootstrap_means.append(np.mean(sample))
            
            # Calculate percentiles for confidence interval
            alpha = (1 - confidence_level) / 2
            lower = np.percentile(bootstrap_means, alpha * 100)
            upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
            
        elif method == "parametric":
            # Parametric method (assumes normal distribution)
            mean = np.mean(data)
            std_error = stats.sem(data)
            h = std_error * stats.norm.ppf((1 + confidence_level) / 2)
            
            lower = mean - h
            upper = mean + h
            
        elif method == "t":
            # t-distribution method (for small samples)
            mean = np.mean(data)
            std_error = stats.sem(data)
            h = std_error * stats.t.ppf((1 + confidence_level) / 2, len(data) - 1)
            
            lower = mean - h
            upper = mean + h
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            "success": True,
            "mean": float(np.mean(data)),
            "lower": float(lower),
            "upper": float(upper),
            "confidence_level": confidence_level,
            "method": method,
            "sample_size": len(data)
        }
        
    except Exception as e:
        logger.error(f"Error calculating confidence intervals: {e}")
        return {
            "success": False,
            "error": str(e),
            "lower": None,
            "upper": None
        }


def validate_model_performance(
    true_values: np.ndarray,
    predicted_values: np.ndarray
) -> Dict[str, float]:
    """
    Validate model performance using various statistical metrics.
    
    Args:
        true_values: Array of actual values
        predicted_values: Array of model-predicted values
        
    Returns:
        Dictionary of performance metrics
    """
    if len(true_values) != len(predicted_values) or len(true_values) < 2:
        return {
            "error": "Invalid input dimensions or insufficient data"
        }
    
    try:
        # Calculate basic performance metrics
        mse = mean_squared_error(true_values, predicted_values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_values, predicted_values)
        r2 = r2_score(true_values, predicted_values)
        
        # Calculate normalized metrics
        if np.std(true_values) > 0:
            nrmse = rmse / np.std(true_values)  # Normalized RMSE
        else:
            nrmse = 0.0
            
        if np.mean(true_values) != 0:
            mape = np.mean(np.abs((true_values - predicted_values) / true_values)) * 100
        else:
            mape = np.nan
        
        # Pearson correlation
        pearson_corr, p_value = stats.pearsonr(true_values, predicted_values)
        
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "normalized_rmse": float(nrmse),
            "mape": float(mape) if not np.isnan(mape) else None,
            "pearson_correlation": float(pearson_corr),
            "p_value": float(p_value),
            "sample_size": len(true_values)
        }
        
    except Exception as e:
        logger.error(f"Error validating model performance: {e}")
        return {
            "error": str(e)
        }


def cross_validate_model(
    model_function: callable,
    features: np.ndarray,
    target: np.ndarray,
    n_splits: int = 5,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform cross-validation on a model to assess performance stability.
    
    Args:
        model_function: Function that takes X_train, y_train and returns a fitted model
        features: Feature matrix
        target: Target values
        n_splits: Number of CV splits
        test_size: Test size for train/test splits
        random_state: Random seed
        
    Returns:
        Dictionary with cross-validation results
    """
    if len(features) != len(target) or len(features) < n_splits * 2:
        return {
            "success": False,
            "error": "Invalid input dimensions or insufficient data for CV"
        }
    
    try:
        # Use KFold for cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Store results for each fold
        fold_results = []
        
        for i, (train_idx, test_idx) in enumerate(kf.split(features)):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = target[train_idx], target[test_idx]
            
            # Train model
            model = model_function(X_train, y_train)
            
            # Generate predictions
            y_pred = model.predict(X_test)
            
            # Calculate performance metrics
            fold_metrics = validate_model_performance(y_test, y_pred)
            fold_metrics["fold"] = i + 1
            
            fold_results.append(fold_metrics)
        
        # Calculate aggregate statistics
        metric_names = [k for k in fold_results[0].keys() if k not in ["fold", "error"]]
        
        aggregate_metrics = {}
        for metric in metric_names:
            values = [fold[metric] for fold in fold_results if metric in fold]
            values = [v for v in values if v is not None]  # Filter out None values
            
            if values:
                aggregate_metrics[f"{metric}_mean"] = float(np.mean(values))
                aggregate_metrics[f"{metric}_std"] = float(np.std(values))
                aggregate_metrics[f"{metric}_min"] = float(np.min(values))
                aggregate_metrics[f"{metric}_max"] = float(np.max(values))
        
        return {
            "success": True,
            "fold_results": fold_results,
            "aggregate_metrics": aggregate_metrics,
            "n_splits": n_splits,
            "sample_size": len(features)
        }
        
    except Exception as e:
        logger.error(f"Error in cross-validation: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def perform_hypothesis_test(
    group1: np.ndarray,
    group2: np.ndarray,
    test_type: str = "t_test",
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform statistical hypothesis test on engagement data.
    
    Args:
        group1: First data sample
        group2: Second data sample
        test_type: Type of test to perform (t_test, wilcoxon, mannwhitney)
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    if len(group1) < 2 or len(group2) < 2:
        return {
            "success": False,
            "error": "Insufficient data for hypothesis testing"
        }
    
    try:
        result = {
            "success": True,
            "test_type": test_type,
            "alpha": alpha,
            "group1_mean": float(np.mean(group1)),
            "group2_mean": float(np.mean(group2)),
            "group1_size": len(group1),
            "group2_size": len(group2)
        }
        
        # Perform the appropriate test
        if test_type == "t_test":
            # Independent samples t-test
            t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
            effect_size = (np.mean(group1) - np.mean(group2)) / np.sqrt(
                (np.std(group1) ** 2 + np.std(group2) ** 2) / 2
            )
            
            result["test_name"] = "Welch's t-test"
            result["t_statistic"] = float(t_stat)
            result["cohen_d"] = float(effect_size)
            
        elif test_type == "wilcoxon":
            # Wilcoxon signed-rank test (paired samples)
            if len(group1) != len(group2):
                raise ValueError("Groups must be the same size for Wilcoxon signed-rank test")
                
            w_stat, p_value = stats.wilcoxon(group1, group2)
            
            # Calculate effect size (r)
            effect_size = abs(w_stat) / np.sqrt(len(group1) * (len(group1) + 1) / 2)
            
            result["test_name"] = "Wilcoxon signed-rank test"
            result["w_statistic"] = float(w_stat)
            result["effect_size_r"] = float(effect_size)
            
        elif test_type == "mannwhitney":
            # Mann-Whitney U test (independent samples, non-parametric)
            u_stat, p_value = stats.mannwhitneyu(group1, group2)
            
            # Calculate effect size (r)
            effect_size = u_stat / (len(group1) * len(group2))
            
            result["test_name"] = "Mann-Whitney U test"
            result["u_statistic"] = float(u_stat)
            result["effect_size_r"] = float(effect_size)
            
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Add p-value and significance result
        result["p_value"] = float(p_value)
        result["significant"] = p_value < alpha
        
        # Add effect size interpretation
        if "cohen_d" in result:
            d = abs(result["cohen_d"])
            if d < 0.2:
                effect_interp = "negligible"
            elif d < 0.5:
                effect_interp = "small"
            elif d < 0.8:
                effect_interp = "medium"
            else:
                effect_interp = "large"
                
            result["effect_size_interpretation"] = effect_interp
            
        elif "effect_size_r" in result:
            r = abs(result["effect_size_r"])
            if r < 0.1:
                effect_interp = "negligible"
            elif r < 0.3:
                effect_interp = "small"
            elif r < 0.5:
                effect_interp = "medium"
            else:
                effect_interp = "large"
                
            result["effect_size_interpretation"] = effect_interp
        
        return result
        
    except Exception as e:
        logger.error(f"Error in hypothesis testing: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def perform_correlation_analysis(
    data: pd.DataFrame,
    features: List[str],
    target: str,
    method: str = "pearson"
) -> Dict[str, Any]:
    """
    Analyze correlations between features and the target variable.
    
    Args:
        data: DataFrame containing features and target
        features: List of feature column names
        target: Name of target column
        method: Correlation method ('pearson', 'spearman', or 'kendall')
        
    Returns:
        Dictionary with correlation analysis
    """
    if not all(col in data.columns for col in features + [target]):
        missing = [col for col in features + [target] if col not in data.columns]
        return {
            "success": False,
            "error": f"Columns not found in data: {missing}"
        }
    
    try:
        correlation_results = []
        
        # Calculate correlations for each feature
        for feature in features:
            # Filter out non-numeric data
            valid_data = data[[feature, target]].dropna()
            
            # Skip if not enough data
            if len(valid_data) < 3:
                correlation_results.append({
                    "feature": feature,
                    "correlation": None,
                    "p_value": None,
                    "sample_size": len(valid_data),
                    "error": "Insufficient data"
                })
                continue
            
            # Calculate correlation based on specified method
            if method == "pearson":
                corr, p_value = stats.pearsonr(valid_data[feature], valid_data[target])
            elif method == "spearman":
                corr, p_value = stats.spearmanr(valid_data[feature], valid_data[target])
            elif method == "kendall":
                corr, p_value = stats.kendalltau(valid_data[feature], valid_data[target])
            else:
                raise ValueError(f"Unknown correlation method: {method}")
            
            # Determine correlation strength interpretation
            abs_corr = abs(corr)
            if abs_corr < 0.1:
                strength = "negligible"
            elif abs_corr < 0.3:
                strength = "weak"
            elif abs_corr < 0.5:
                strength = "moderate"
            elif abs_corr < 0.7:
                strength = "strong"
            else:
                strength = "very strong"
            
            # Determine direction
            direction = "positive" if corr > 0 else "negative" if corr < 0 else "neutral"
            
            # Determine significance
            significant = p_value < 0.05
            
            correlation_results.append({
                "feature": feature,
                "correlation": float(corr),
                "p_value": float(p_value),
                "sample_size": len(valid_data),
                "strength": strength,
                "direction": direction,
                "significant": significant
            })
        
        # Sort by absolute correlation strength
        correlation_results.sort(key=lambda x: abs(x["correlation"]) if x["correlation"] is not None else -1, reverse=True)
        
        # Calculate basic statistics
        significant_correlations = [r for r in correlation_results if r["significant"] and r["correlation"] is not None]
        
        return {
            "success": True,
            "method": method,
            "target": target,
            "correlations": correlation_results,
            "sig_correlation_count": len(significant_correlations),
            "total_features": len(features)
        }
        
    except Exception as e:
        logger.error(f"Error in correlation analysis: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def validate_dimension_reliability(
    dimension_scores: Dict[str, List[float]],
    min_reliability: float = 0.7
) -> Dict[str, Any]:
    """
    Validate the reliability of engagement dimensions using Cronbach's alpha.
    
    Args:
        dimension_scores: Dictionary mapping dimension names to lists of scores
        min_reliability: Minimum acceptable reliability coefficient
        
    Returns:
        Dictionary with reliability analysis
    """
    if not dimension_scores:
        return {
            "success": False,
            "error": "No dimension scores provided"
        }
    
    try:
        reliability_results = {}
        
        for dimension, scores in dimension_scores.items():
            # Convert scores to DataFrame for analysis
            scores_array = np.array(scores)
            
            # Need at least 3 items and 10 observations for meaningful analysis
            if scores_array.shape[1] < 3:
                reliability_results[dimension] = {
                    "reliability": None,
                    "interpretation": "insufficient_items",
                    "items_count": scores_array.shape[1],
                    "observations_count": scores_array.shape[0]
                }
                continue
                
            if scores_array.shape[0] < 10:
                reliability_results[dimension] = {
                    "reliability": None,
                    "interpretation": "insufficient_observations",
                    "items_count": scores_array.shape[1],
                    "observations_count": scores_array.shape[0]
                }
                continue
            
            # Calculate Cronbach's alpha
            item_variances = np.var(scores_array, axis=0, ddof=1)
            total_variance = np.var(np.sum(scores_array, axis=1), ddof=1)
            
            n_items = scores_array.shape[1]
            cronbach_alpha = (n_items / (n_items - 1)) * (1 - np.sum(item_variances) / total_variance)
            
            # Interpret reliability
            if cronbach_alpha >= 0.9:
                reliability_interp = "excellent"
            elif cronbach_alpha >= 0.8:
                reliability_interp = "good"
            elif cronbach_alpha >= 0.7:
                reliability_interp = "acceptable"
            elif cronbach_alpha >= 0.6:
                reliability_interp = "questionable"
            elif cronbach_alpha >= 0.5:
                reliability_interp = "poor"
            else:
                reliability_interp = "unacceptable"
            
            reliability_results[dimension] = {
                "reliability": float(cronbach_alpha),
                "interpretation": reliability_interp,
                "meets_threshold": cronbach_alpha >= min_reliability,
                "items_count": n_items,
                "observations_count": scores_array.shape[0]
            }
        
        # Summarize overall reliability
        dimensions_count = len(reliability_results)
        reliable_dimensions = sum(1 for dim_result in reliability_results.values() 
                               if dim_result.get("meets_threshold", False))
        
        return {
            "success": True,
            "dimension_reliability": reliability_results,
            "reliable_dimensions_count": reliable_dimensions,
            "total_dimensions": dimensions_count,
            "reliability_threshold": min_reliability
        }
        
    except Exception as e:
        logger.error(f"Error in reliability analysis: {e}")
        return {
            "success": False,
            "error": str(e)
        } 