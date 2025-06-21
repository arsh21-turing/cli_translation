"""
Builds and trains predictive models for quality assessment.
"""
import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Configure logging using unified configuration
from logger_config import get_logger
logger = get_logger("PredictionModelBuilder", "quality_learning")

class PredictionModelBuilder:
    """
    Builds and trains predictive models for quality assessment.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the prediction model builder.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def prepare_dataset(
        self, 
        data: pd.DataFrame, 
        target_column: str, 
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare train/test datasets for model training.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column to predict
            test_size: Fraction of data to use for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if target_column not in data.columns:
            raise ValueError(f"Target column {target_column} not in data")
            
        # Extract features and target
        numeric_data = data.select_dtypes(include=['number'])
        
        if target_column not in numeric_data.columns:
            raise ValueError(f"Target column {target_column} is not numeric")
            
        X = numeric_data.drop(columns=[target_column])
        y = numeric_data[target_column]
        
        # Remove any other non-feature columns
        exclude_columns = ['file_path', 'timestamp', 'id', 'index']
        for col in exclude_columns:
            if col in X.columns:
                X = X.drop(columns=[col])
                
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        logger.info(f"Prepared dataset with {len(X_train)} training samples and {len(X_test)} test samples")
        logger.info(f"Features: {list(X_train.columns)}")
        
        return X_train, X_test, y_train, y_test
    
    def build_linear_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """
        Build and train a linear regression model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained model
        """
        # Create pipeline with scaling
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        
        # Train model
        model.fit(X_train, y_train)
        
        # Get coefficients
        coefficients = model.named_steps['regressor'].coef_
        feature_importances = dict(zip(X_train.columns, coefficients))
        
        # Log most important features
        sorted_features = sorted(
            feature_importances.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        logger.info("Trained linear regression model")
        logger.info("Top coefficients:")
        for feature, coef in sorted_features[:5]:
            logger.info(f"  {feature}: {coef:.4f}")
            
        return model
    
    def build_random_forest_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """
        Build and train a random forest regression model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained model
        """
        # Create pipeline with scaling
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(
                n_estimators=100, 
                max_depth=None, 
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=self.random_state
            ))
        ])
        
        # Train model
        model.fit(X_train, y_train)
        
        # Get feature importances
        importances = model.named_steps['regressor'].feature_importances_
        feature_importances = dict(zip(X_train.columns, importances))
        
        # Log most important features
        sorted_features = sorted(
            feature_importances.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        logger.info("Trained random forest regression model")
        logger.info("Top feature importances:")
        for feature, importance in sorted_features[:5]:
            logger.info(f"  {feature}: {importance:.4f}")
            
        return model
    
    def build_gradient_boosting_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """
        Build and train a gradient boosting regression model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained model
        """
        # Create pipeline with scaling
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=self.random_state
            ))
        ])
        
        # Train model
        model.fit(X_train, y_train)
        
        # Get feature importances
        importances = model.named_steps['regressor'].feature_importances_
        feature_importances = dict(zip(X_train.columns, importances))
        
        # Log most important features
        sorted_features = sorted(
            feature_importances.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        logger.info("Trained gradient boosting regression model")
        logger.info("Top feature importances:")
        for feature, importance in sorted_features[:5]:
            logger.info(f"  {feature}: {importance:.4f}")
            
        return model
    
    def build_neural_network_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """
        Build and train a neural network regression model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained model
        """
        # Determine hidden layer sizes based on input dimensions
        n_features = X_train.shape[1]
        hidden_layer_sizes = (max(n_features * 2, 10), max(n_features, 5))
        
        # Create pipeline with scaling
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size='auto',
                learning_rate='constant',
                learning_rate_init=0.001,
                max_iter=1000,
                random_state=self.random_state
            ))
        ])
        
        # Train model
        model.fit(X_train, y_train)
        
        logger.info("Trained neural network regression model")
        logger.info(f"Network architecture: {n_features} input -> {hidden_layer_sizes[0]} -> {hidden_layer_sizes[1]} -> 1 output")
            
        return model
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            model: Model to evaluate
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate additional metrics
        explained_variance = 1 - np.var(y_test - y_pred) / np.var(y_test)
        
        # Calculate residuals
        residuals = y_test - y_pred
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2),
            "explained_variance": float(explained_variance),
            "mean_residual": float(mean_residual),
            "std_residual": float(std_residual)
        }
        
        logger.info(f"Model evaluation metrics: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R^2={r2:.4f}")
        
        return metrics
    
    def cross_validate_model(self, model: Any, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """
        Cross-validate model performance.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Targets
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation metrics
        """
        # Calculate cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        
        r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        
        # Compile metrics
        cv_metrics = {
            "mean_rmse": float(cv_rmse.mean()),
            "std_rmse": float(cv_rmse.std()),
            "mean_r2": float(r2_scores.mean()),
            "std_r2": float(r2_scores.std()),
            "cv_folds": cv
        }
        
        logger.info(f"Cross-validation results ({cv} folds): RMSE={cv_rmse.mean():.4f}±{cv_rmse.std():.4f}, R^2={r2_scores.mean():.4f}±{r2_scores.std():.4f}")
        
        return cv_metrics
    
    def get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """
        Get importance of features from a trained model.
        
        Args:
            model: Trained model
            feature_names: Names of features
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Check if model is a pipeline
        if hasattr(model, 'named_steps') and 'regressor' in model.named_steps:
            regressor = model.named_steps['regressor']
        else:
            regressor = model
            
        # Extract feature importances based on model type
        importances = {}
        
        if hasattr(regressor, 'feature_importances_'):
            # Tree-based models
            importances = dict(zip(feature_names, regressor.feature_importances_))
        elif hasattr(regressor, 'coef_'):
            # Linear models
            if len(regressor.coef_.shape) > 1:
                importances = dict(zip(feature_names, np.abs(regressor.coef_[0])))
            else:
                importances = dict(zip(feature_names, np.abs(regressor.coef_)))
        else:
            logger.warning("Model doesn't expose feature importances")
            return {}
            
        # Sort importances
        sorted_importances = dict(sorted(
            importances.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return sorted_importances
    
    def predict_quality(self, model: Any, features: pd.DataFrame) -> np.ndarray:
        """
        Predict quality with model.
        
        Args:
            model: Trained model
            features: Features to predict from
            
        Returns:
            Array of predictions
        """
        return model.predict(features)
    
    def optimize_hyperparameters(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        model_type: str
    ) -> Dict[str, Any]:
        """
        Optimize model hyperparameters.
        
        Args:
            X: Features
            y: Targets
            model_type: Type of model to optimize
            
        Returns:
            Dictionary with optimized model and parameters
        """
        if model_type == "linear":
            # Linear models have limited hyperparameters, return default
            model = self.build_linear_model(X, y)
            return {
                "model": model,
                "best_params": {},
                "best_score": float(np.mean(cross_val_score(model, X, y, cv=5, scoring='r2')))
            }
        
        elif model_type == "random_forest":
            # Define parameter grid
            param_grid = {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [None, 5, 10, 20],
                'regressor__min_samples_split': [2, 5, 10],
                'regressor__min_samples_leaf': [1, 2, 4]
            }
            
            # Create base model
            base_model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', RandomForestRegressor(random_state=self.random_state))
            ])
            
        elif model_type == "gradient_boosting":
            # Define parameter grid
            param_grid = {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
                'regressor__max_depth': [3, 5, 7, 9],
                'regressor__min_samples_split': [2, 5, 10]
            }
            
            # Create base model
            base_model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', GradientBoostingRegressor(random_state=self.random_state))
            ])
            
        elif model_type == "neural_network":
            # Define parameter grid
            n_features = X.shape[1]
            param_grid = {
                'regressor__hidden_layer_sizes': [
                    (n_features,), 
                    (n_features*2,), 
                    (n_features, n_features//2),
                    (n_features*2, n_features)
                ],
                'regressor__alpha': [0.0001, 0.001, 0.01],
                'regressor__learning_rate_init': [0.001, 0.01, 0.1],
                'regressor__max_iter': [500, 1000]
            }
            
            # Create base model
            base_model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', MLPRegressor(
                    solver='adam',
                    activation='relu',
                    random_state=self.random_state
                ))
            ])
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Perform grid search
        logger.info(f"Optimizing hyperparameters for {model_type} model...")
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        # Get best model and parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best cross-validation R^2: {best_score:.4f}")
        
        return {
            "model": best_model,
            "best_params": best_params,
            "best_score": float(best_score)
        }
    
    def calibrate_model_predictions(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Any:
        """
        Calibrate probability predictions of a model.
        
        Args:
            model: Trained model
            X: Calibration features
            y: Calibration targets
            
        Returns:
            Calibrated model
        """
        # For regression models, we can't use the standard calibration
        # Instead, we'll use a simple adjustment based on the residuals
        
        # Get predictions
        y_pred = model.predict(X)
        
        # Calculate mean error
        mean_error = np.mean(y - y_pred)
        
        logger.info(f"Calibrating model with mean error adjustment: {mean_error:.4f}")
        
        # Create a wrapped model that applies the adjustment
        class CalibratedModel:
            def __init__(self, base_model, adjustment):
                self.base_model = base_model
                self.adjustment = adjustment
                
            def predict(self, X):
                return self.base_model.predict(X) + self.adjustment
                
            # Pass through any attributes to the base model
            def __getattr__(self, name):
                return getattr(self.base_model, name)
        
        return CalibratedModel(model, mean_error)
    
    def save_model(self, model: Any, path: str) -> None:
        """
        Save model to file.
        
        Args:
            model: Model to save
            path: Path to save model to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        with open(path, 'wb') as f:
            pickle.dump(model, f)
            
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> Any:
        """
        Load model from file.
        
        Args:
            path: Path to load model from
            
        Returns:
            Loaded model
        """
        # Check if file exists
        if not os.path.exists(path):
            raise ValueError(f"Model file not found: {path}")
            
        # Load model
        with open(path, 'rb') as f:
            model = pickle.load(f)
            
        logger.info(f"Model loaded from {path}")
        
        return model 