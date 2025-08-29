import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pickle
import os
from datetime import datetime

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, 
                             GradientBoostingClassifier, GradientBoostingRegressor,
                             AdaBoostClassifier, AdaBoostRegressor)
from sklearn.linear_model import (LogisticRegression, LinearRegression, Ridge, Lasso, 
                                 ElasticNet)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           mean_squared_error, r2_score, mean_absolute_error,
                           roc_auc_score, precision_recall_curve, roc_curve,
                           f1_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from scipy.stats import median_abs_deviation
import warnings
warnings.filterwarnings('ignore')

class AdvancedML:
    """Advanced ML utilities class"""
    
    def __init__(self, data):
        self.data = data.copy()
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.is_classification = True
        self.scaler = None
        self.label_encoders = {}
        
    def advanced_preprocessing(self):
        """Advanced data preprocessing options"""
        st.subheader("🔧 Advanced Data Preprocessing")
        
        preprocessing_options = st.multiselect(
            "Select preprocessing steps:",
            ["Handle Missing Values", "Feature Scaling", "Feature Selection", 
             "Outlier Handling", "Feature Engineering"],
            default=["Handle Missing Values"]
        )
        
        processed_data = self.data.copy()
        
        if "Handle Missing Values" in preprocessing_options:
            processed_data = self._handle_missing_values(processed_data)
            
        if "Feature Scaling" in preprocessing_options:
            processed_data = self._feature_scaling(processed_data)
            
        if "Outlier Handling" in preprocessing_options:
            processed_data = self._handle_outliers(processed_data)
            
        if "Feature Engineering" in preprocessing_options:
            processed_data = self._feature_engineering(processed_data)
            
        return processed_data
    
    def _handle_missing_values(self, data):
        """Advanced missing value handling"""
        st.write("**Missing Value Treatment:**")
        
        # Show missing value summary
        missing_summary = data.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0].index.tolist()
        
        if not missing_cols:
            st.success("No missing values found!")
            return data
            
        strategy_mapping = {}
        
        for col in missing_cols:
            col_type = data[col].dtype
            if col_type in ['int64', 'float64']:
                options = ["mean", "median", "most_frequent", "constant", "drop"]
            else:
                options = ["most_frequent", "constant", "drop"]
                
            strategy = st.selectbox(f"Strategy for {col}:", options, key=f"strategy_{col}")
            strategy_mapping[col] = strategy
            
            if strategy == "constant":
                fill_value = st.text_input(f"Fill value for {col}:", "0", key=f"fill_{col}")
                strategy_mapping[f"{col}_fill"] = fill_value
        
        # Apply strategies
        for col, strategy in strategy_mapping.items():
            if col.endswith('_fill'):
                continue
                
            if strategy == "drop":
                data = data.dropna(subset=[col])
            elif strategy == "constant":
                fill_value = strategy_mapping.get(f"{col}_fill", "0")
                if data[col].dtype in ['int64', 'float64']:
                    try:
                        fill_value = float(fill_value)
                    except:
                        fill_value = 0
                data[col] = data[col].fillna(fill_value)
            else:
                if data[col].dtype in ['int64', 'float64']:
                    imputer = SimpleImputer(strategy=strategy)
                    data[col] = imputer.fit_transform(data[[col]]).ravel()
                else:
                    imputer = SimpleImputer(strategy=strategy)
                    data[col] = imputer.fit_transform(data[[col]]).ravel()
        
        return data
    
    def _feature_scaling(self, data):
        """Apply feature scaling"""
        st.write("**Feature Scaling:**")
        
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numerical_cols:
            st.warning("No numerical columns found for scaling.")
            return data
            
        scaling_method = st.selectbox("Select scaling method:", 
                                    ["StandardScaler", "MinMaxScaler", "None"])
        
        if scaling_method != "None":
            cols_to_scale = st.multiselect("Select columns to scale:", 
                                         numerical_cols, default=numerical_cols)
            
            if cols_to_scale:
                if scaling_method == "StandardScaler":
                    scaler = StandardScaler()
                else:
                    scaler = MinMaxScaler()
                    
                data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])
                self.scaler = scaler  # Store for later use
                st.success(f"Applied {scaling_method} to {len(cols_to_scale)} columns")
        
        return data
    
    def _handle_outliers(self, data):
        """Handle outliers"""
        st.write("**Outlier Handling:**")
        
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numerical_cols:
            st.warning("No numerical columns found for outlier handling.")
            return data
            
        outlier_method = st.selectbox("Select outlier handling method:", 
                                    ["Remove (IQR)", "Cap (IQR)", "None"])
        
        if outlier_method != "None":
            cols_to_process = st.multiselect("Select columns for outlier handling:", 
                                           numerical_cols)
            
            for col in cols_to_process:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                if outlier_method == "Remove (IQR)":
                    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
                elif outlier_method == "Cap (IQR)":
                    data[col] = np.clip(data[col], lower_bound, upper_bound)
        
        return data
    
    def _feature_engineering(self, data):
        """Basic feature engineering"""
        st.write("**Feature Engineering:**")
        
        engineering_options = st.multiselect(
            "Select feature engineering options:",
            ["Polynomial Features", "Interaction Features", "Binning"]
        )
        
        # This is a simplified version - you can expand based on needs
        if engineering_options:
            st.info("Feature engineering options selected. Implementation can be customized based on specific needs.")
        
        return data
    
    def hyperparameter_tuning(self, X_train, y_train, model_name, model):
        """Perform hyperparameter tuning"""
        st.subheader("🎯 Hyperparameter Tuning")
        
        tuning_method = st.selectbox("Select tuning method:", 
                                   ["Grid Search", "Random Search", "None"])
        
        if tuning_method == "None":
            return model
            
        # Define parameter grids for common models
        param_grids = self._get_parameter_grids()
        
        if model_name not in param_grids:
            st.warning(f"No parameter grid defined for {model_name}")
            return model
            
        param_grid = param_grids[model_name]
        
        # Allow user to modify parameters
        st.write("**Parameter Grid:**")
        for param, values in param_grid.items():
            st.write(f"- {param}: {values}")
        
        if st.button(f"Start {tuning_method}"):
            with st.spinner(f"Running {tuning_method}..."):
                if tuning_method == "Grid Search":
                    search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy' if self.is_classification else 'r2')
                else:
                    search = RandomizedSearchCV(model, param_grid, cv=5, n_iter=20, scoring='accuracy' if self.is_classification else 'r2', random_state=42)
                
                search.fit(X_train, y_train)
                
                st.success("Hyperparameter tuning completed!")
                st.write(f"**Best Parameters:** {search.best_params_}")
                st.write(f"**Best Score:** {search.best_score_:.4f}")
                
                return search.best_estimator_
        
        return model
    
    def _get_parameter_grids(self):
        """Define parameter grids for hyperparameter tuning"""
        return {
            'RandomForestClassifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'RandomForestRegressor': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'LogisticRegression': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'SVC': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            'GradientBoostingClassifier': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
    
    def feature_selection(self, X, y):
        """Perform feature selection"""
        st.subheader("🎯 Feature Selection")
        
        if X.shape[1] < 2:
            st.warning("Need at least 2 features for selection.")
            return X
            
        selection_method = st.selectbox("Select feature selection method:",
                                      ["SelectKBest", "RFE", "None"])
        
        if selection_method == "None":
            return X
            
        if selection_method == "SelectKBest":
            k = st.slider("Number of features to select:", 1, X.shape[1], min(10, X.shape[1]))
            
            if self.is_classification:
                selector = SelectKBest(score_func=f_classif, k=k)
            else:
                selector = SelectKBest(score_func=f_regression, k=k)
                
            X_selected = selector.fit_transform(X, y)
            
            # Show selected features
            selected_features = X.columns[selector.get_support()].tolist()
            st.write(f"**Selected Features:** {selected_features}")
            
            return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            
        elif selection_method == "RFE":
            n_features = st.slider("Number of features to select:", 1, X.shape[1], min(5, X.shape[1]))
            
            if self.is_classification:
                estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
                
            selector = RFE(estimator, n_features_to_select=n_features)
            X_selected = selector.fit_transform(X, y)
            
            # Show selected features
            selected_features = X.columns[selector.get_support()].tolist()
            st.write(f"**Selected Features:** {selected_features}")
            
            return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        return X
    
    def cross_validation_analysis(self, model, X, y):
        """Perform cross-validation analysis"""
        st.subheader("🔄 Cross-Validation Analysis")
        
        cv_folds = st.slider("Number of CV folds:", 3, 10, 5)
        
        if st.button("Run Cross-Validation"):
            with st.spinner("Running cross-validation..."):
                scoring = 'accuracy' if self.is_classification else 'r2'
                cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean CV Score", f"{cv_scores.mean():.4f}")
                with col2:
                    st.metric("Std CV Score", f"{cv_scores.std():.4f}")
                with col3:
                    st.metric("Best Fold Score", f"{cv_scores.max():.4f}")
                
                # Visualize CV scores
                fig = px.bar(x=[f"Fold {i+1}" for i in range(len(cv_scores))], 
                           y=cv_scores,
                           title="Cross-Validation Scores by Fold")
                fig.add_hline(y=cv_scores.mean(), line_dash="dash", 
                            annotation_text=f"Mean: {cv_scores.mean():.4f}")
                st.plotly_chart(fig, use_container_width=True)
    
    def model_interpretation(self, model, X, feature_names):
        """Provide model interpretation"""
        st.subheader("🔍 Model Interpretation")
        
        # Feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            st.write("**Feature Importance:**")
            
            importances = model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Plot feature importance
            fig = px.bar(feature_importance.head(10), 
                        x='Importance', y='Feature', orientation='h',
                        title='Top 10 Feature Importances')
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(feature_importance, use_container_width=True)
        
        # Coefficients for linear models
        elif hasattr(model, 'coef_'):
            st.write("**Model Coefficients:**")
            
            if self.is_classification and len(model.coef_.shape) > 1:
                # Multi-class classification
                coef_df = pd.DataFrame(model.coef_.T, 
                                     index=feature_names,
                                     columns=[f'Class_{i}' for i in range(model.coef_.shape[0])])
            else:
                # Binary classification or regression
                coef = model.coef_.flatten() if len(model.coef_.shape) > 1 else model.coef_
                coef_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': coef
                }).sort_values('Coefficient', key=abs, ascending=False)
                
                # Plot coefficients
                fig = px.bar(coef_df.head(10), 
                           x='Coefficient', y='Feature', orientation='h',
                           title='Top 10 Feature Coefficients (by absolute value)')
                st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(coef_df, use_container_width=True)
        
        else:
            st.info("Feature importance/coefficients not available for this model type.")
    
    def advanced_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate and display advanced metrics"""
        st.subheader("📊 Advanced Model Metrics")
        
        if self.is_classification:
            self._classification_metrics(y_true, y_pred, y_pred_proba)
        else:
            self._regression_metrics(y_true, y_pred)
    
    def _classification_metrics(self, y_true, y_pred, y_pred_proba):
        """Advanced classification metrics"""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
        with col2:
            st.metric("Precision", f"{precision:.4f}")
        with col3:
            st.metric("Recall", f"{recall:.4f}")
        with col4:
            st.metric("F1-Score", f"{f1:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        fig = px.imshow(cm, text_auto=True, aspect="auto",
                       title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
        # ROC Curve and AUC (for binary classification)
        if len(np.unique(y_true)) == 2 and y_pred_proba is not None:
            st.write("**ROC Curve:**")
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            
            fig = px.line(x=fpr, y=tpr, title=f'ROC Curve (AUC = {auc:.4f})')
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                         line=dict(dash="dash", color="gray"))
            fig.update_xaxes(title='False Positive Rate')
            fig.update_yaxes(title='True Positive Rate')
            st.plotly_chart(fig, use_container_width=True)
            
            # Precision-Recall Curve
            st.write("**Precision-Recall Curve:**")
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
            
            fig = px.line(x=recall_curve, y=precision_curve, 
                         title='Precision-Recall Curve')
            fig.update_xaxes(title='Recall')
            fig.update_yaxes(title='Precision')
            st.plotly_chart(fig, use_container_width=True)
        
        # Classification Report
        st.write("**Detailed Classification Report:**")
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
    
    def _regression_metrics(self, y_true, y_pred):
        """Advanced regression metrics"""
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MSE", f"{mse:.4f}")
        with col2:
            st.metric("RMSE", f"{rmse:.4f}")
        with col3:
            st.metric("MAE", f"{mae:.4f}")
        with col4:
            st.metric("R² Score", f"{r2:.4f}")
        
        # Residual plots
        residuals = y_true - y_pred
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Actual vs Predicted
            fig = px.scatter(x=y_true, y=y_pred, 
                           title="Actual vs Predicted Values")
            fig.add_shape(type="line", x0=min(y_true), y0=min(y_true),
                         x1=max(y_true), y1=max(y_true),
                         line=dict(dash="dash", color="red"))
            fig.update_xaxes(title='Actual')
            fig.update_yaxes(title='Predicted')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Residual plot
            fig = px.scatter(x=y_pred, y=residuals,
                           title="Residual Plot")
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_xaxes(title='Predicted')
            fig.update_yaxes(title='Residuals')
            st.plotly_chart(fig, use_container_width=True)
        
        # Residual distribution
        fig = px.histogram(residuals, title="Distribution of Residuals")
        st.plotly_chart(fig, use_container_width=True)
    
    def save_model(self, model, model_name):
        """Save trained model"""
        st.subheader("💾 Save Model")
        
        if not os.path.exists('models'):
            os.makedirs('models')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.pkl"
        filepath = os.path.join('models', filename)
        
        if st.button("Save Model"):
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump({
                        'model': model,
                        'scaler': self.scaler,
                        'label_encoders': self.label_encoders,
                        'timestamp': timestamp,
                        'model_name': model_name
                    }, f)
                st.success(f"Model saved as {filename}")
                
                # Also save model metadata
                metadata = {
                    'filename': filename,
                    'model_name': model_name,
                    'timestamp': timestamp,
                    'is_classification': self.is_classification
                }
                
                metadata_file = os.path.join('models', 'model_registry.txt')
                with open(metadata_file, 'a') as f:
                    f.write(f"{metadata}\n")
                    
            except Exception as e:
                st.error(f"Error saving model: {str(e)}")
    
    def load_model(self):
        """Load a saved model"""
        st.subheader("📁 Load Model")
        
        if not os.path.exists('models'):
            st.warning("No models directory found.")
            return None
            
        model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
        
        if not model_files:
            st.warning("No saved models found.")
            return None
            
        selected_model = st.selectbox("Select model to load:", model_files)
        
        if st.button("Load Model"):
            try:
                filepath = os.path.join('models', selected_model)
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                
                st.success(f"Model {selected_model} loaded successfully!")
                return model_data
                
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                return None
        
        return None
    
    def batch_prediction(self, model, scaler=None):
        """Perform batch predictions on new data"""
        st.subheader("🔮 Batch Prediction")
        
        uploaded_file = st.file_uploader("Upload CSV for prediction", type="csv")
        
        if uploaded_file is not None:
            try:
                new_data = pd.read_csv(uploaded_file)
                st.write("**Data Preview:**")
                st.dataframe(new_data.head(), use_container_width=True)
                
                # Prepare data (same preprocessing as training)
                processed_data = new_data.copy()
                
                # Apply same preprocessing steps
                if scaler is not None:
                    numerical_cols = processed_data.select_dtypes(include=[np.number]).columns
                    processed_data[numerical_cols] = scaler.transform(processed_data[numerical_cols])
                
                # Make predictions
                predictions = model.predict(processed_data)
                
                # Add predictions to the dataframe
                result_data = new_data.copy()
                result_data['Predictions'] = predictions
                
                if hasattr(model, 'predict_proba') and self.is_classification:
                    probabilities = model.predict_proba(processed_data)
                    for i in range(probabilities.shape[1]):
                        result_data[f'Probability_Class_{i}'] = probabilities[:, i]
                
                st.write("**Predictions:**")
                st.dataframe(result_data, use_container_width=True)
                
                # Download option
                csv = result_data.to_csv(index=False)
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error in batch prediction: {str(e)}")
    
    def ensemble_methods(self, X_train, y_train, X_test, y_test):
        """Implement ensemble methods"""
        st.subheader("🎭 Ensemble Methods")
        
        ensemble_type = st.selectbox("Select ensemble method:",
                                   ["Voting", "Bagging", "Boosting"])
        
        if ensemble_type == "Voting":
            self._voting_ensemble(X_train, y_train, X_test, y_test)
        elif ensemble_type == "Bagging":
            self._bagging_ensemble(X_train, y_train, X_test, y_test)
        else:
            self._boosting_ensemble(X_train, y_train, X_test, y_test)
    
    def _voting_ensemble(self, X_train, y_train, X_test, y_test):
        """Implement voting ensemble"""
        from sklearn.ensemble import VotingClassifier, VotingRegressor
        
        if self.is_classification:
            base_models = [
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                ('lr', LogisticRegression(random_state=42)),
                ('svm', SVC(probability=True, random_state=42))
            ]
            ensemble = VotingClassifier(estimators=base_models, voting='soft')
        else:
            base_models = [
                ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
                ('lr', LinearRegression()),
                ('svr', SVR())
            ]
            ensemble = VotingRegressor(estimators=base_models)
        
        if st.button("Train Voting Ensemble"):
            with st.spinner("Training ensemble..."):
                ensemble.fit(X_train, y_train)
                y_pred = ensemble.predict(X_test)
                
                if self.is_classification:
                    accuracy = accuracy_score(y_test, y_pred)
                    st.metric("Ensemble Accuracy", f"{accuracy:.4f}")
                else:
                    r2 = r2_score(y_test, y_pred)
                    st.metric("Ensemble R²", f"{r2:.4f}")
    
    def _bagging_ensemble(self, X_train, y_train, X_test, y_test):
        """Implement bagging ensemble"""
        from sklearn.ensemble import BaggingClassifier, BaggingRegressor
        
        n_estimators = st.slider("Number of estimators:", 10, 100, 50)
        
        if self.is_classification:
            ensemble = BaggingClassifier(n_estimators=n_estimators, random_state=42)
        else:
            ensemble = BaggingRegressor(n_estimators=n_estimators, random_state=42)
        
        if st.button("Train Bagging Ensemble"):
            with st.spinner("Training ensemble..."):
                ensemble.fit(X_train, y_train)
                y_pred = ensemble.predict(X_test)
                
                if self.is_classification:
                    accuracy = accuracy_score(y_test, y_pred)
                    st.metric("Ensemble Accuracy", f"{accuracy:.4f}")
                else:
                    r2 = r2_score(y_test, y_pred)
                    st.metric("Ensemble R²", f"{r2:.4f}")
    
    def _boosting_ensemble(self, X_train, y_train, X_test, y_test):
        """Implement boosting ensemble"""
        n_estimators = st.slider("Number of estimators:", 50, 200, 100)
        learning_rate = st.slider("Learning rate:", 0.01, 0.3, 0.1)
        
        if self.is_classification:
            ensemble = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=42
            )
        else:
            ensemble = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=42
            )
        
        if st.button("Train Gradient Boosting Ensemble"):
            with st.spinner("Training ensemble..."):
                ensemble.fit(X_train, y_train)
                y_pred = ensemble.predict(X_test)
                
                if self.is_classification:
                    accuracy = accuracy_score(y_test, y_pred)
                    st.metric("Ensemble Accuracy", f"{accuracy:.4f}")
                else:
                    r2 = r2_score(y_test, y_pred)
                    st.metric("Ensemble R²", f"{r2:.4f}")
                    
def quick_model_preview(df, target_column, selected_features=None, problem_type=None):
    try:
        # Select features
        X = df[selected_features] if selected_features else df.drop(columns=[target_column])
        y = df[target_column]

        # Check if we have enough data
        if len(X) < 10:
            raise ValueError("Not enough data for training (minimum 10 samples required)")

        # 🔹 Encode categorical features in X
        X = X.copy()
        for col in X.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # 🔹 Encode categorical target if classification
        if problem_type == "Classification" and y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))

        # 🔹 Handle missing values (imputation)
        imputer = SimpleImputer(strategy="most_frequent")   # or "mean" for numeric
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        results = {}

        if problem_type == "Classification":
            model = LogisticRegression(max_iter=200)
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            results['performance'] = {"Accuracy": acc}
            results['model'] = model

        elif problem_type == "Regression":
            model = LinearRegression()
            model.fit(X_train, y_train)
            r2 = r2_score(y_test, model.predict(X_test))
            results['performance'] = {"R² Score": r2}
            results['model'] = model

        else:
            return None

        return results
    except Exception as e:
        print(f"Error in quick_model_preview: {str(e)}")
        return None

def feature_importance_analysis(model, feature_names):
    try:
        importances = None

        if hasattr(model, "feature_importances_"):   # Tree-based models
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):  # Linear/Logistic regression
            importances = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        else:
            raise ValueError("⚠️ Model does not support feature importance extraction")

        # Ensure feature_names and importances have the same length
        if len(feature_names) != len(importances):
            min_len = min(len(feature_names), len(importances))
            feature_names = feature_names[:min_len]
            importances = importances[:min_len]

        # Put into DataFrame
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": np.abs(importances)
        }).sort_values(by="Importance", ascending=False)

        # Plot
        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance",
            text="Importance"
        )
        fig.update_traces(texttemplate="%{text:.3f}")
        return fig
    except Exception as e:
        # Return a simple error figure if something goes wrong
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error generating feature importance: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title="Feature Importance - Error")
        return fig
    
def data_preprocessing_suggestions(df: pd.DataFrame):
    suggestions = []

    # Missing values
    if df.isnull().sum().sum() > 0:
        suggestions.append("Handle missing values (e.g., imputation or removal).")

    # Categorical variables
    if (df.dtypes == 'object').any():
        suggestions.append("Encode categorical features (e.g., OneHotEncoder, LabelEncoder).")

    # Duplicates
    if df.duplicated().sum() > 0:
        suggestions.append("Remove duplicate rows to avoid data leakage.")

    # Scaling
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(numeric_cols) > 0:
        suggestions.append("Consider feature scaling (StandardScaler or MinMaxScaler).")

    # Outliers
    if len(numeric_cols) > 0:
        suggestions.append("Check for outliers and handle them (e.g., winsorization, log transform).")

    # Class imbalance
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() == 2:  # binary target candidate
            try:
                imbalance = df[col].value_counts(normalize=True).values
                if len(imbalance) > 0 and float(max(imbalance)) > 0.8:  # imbalance threshold
                    suggestions.append(f"Target column '{col}' seems imbalanced. Consider resampling techniques (SMOTE/undersampling).")
            except Exception as e:
                # Skip this check if there's an error
                continue

    return suggestions