import streamlit as st
import pandas as pd
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from io import BytesIO
import itertools

st.set_page_config(layout="wide")
st.title("Echo State Network Predictor (ESNapp)")

# -----------------------------
# 1️⃣ Upload training file
# -----------------------------
file_uploader = st.file_uploader("Upload Training Data (Excel or CSV)", type=["xlsx", "csv"])
model_trained = False

if file_uploader:
    # Handle both Excel and CSV files
    file_type = file_uploader.name.split('.')[-1]
    if file_type == 'csv':
        train_df = pd.read_csv(file_uploader)
    elif file_type == 'xlsx':
        excel_sheets = pd.ExcelFile(file_uploader).sheet_names
        selected_sheet = st.selectbox("Select the sheet for training data", excel_sheets)
        train_df = pd.read_excel(file_uploader, sheet_name=selected_sheet)
    
    st.subheader("Training Data Preview")
    st.dataframe(train_df.head())

    input_cols_train = st.multiselect(
        "Select input columns for training", 
        train_df.columns.tolist(), 
        default=train_df.columns[:-1].tolist()
    )
    output_cols_train = st.multiselect(
        "Select output column(s) for training", 
        train_df.columns.tolist(), 
        default=[train_df.columns[-1]]
    )

    # -----------------------------
    # 2️⃣ ESN Hyperparameters
    # -----------------------------
    st.subheader("ESN Hyperparameters")
    
    # 2a️⃣ Optional Grid Search
    grid_search = st.checkbox("Use Grid Search for best hyperparameters?")
    if grid_search:
        # User-defined ranges for grid search
        n_reservoir_range = st.text_input("Enter n_reservoir values (e.g., 20,50,100)", "20, 50, 100")
        spectral_radius_range = st.text_input("Enter spectral radius values (e.g., 0.7,0.9,1.1)", "0.7, 0.9, 1.1")
        
        try:
            reservoir_options = [int(x.strip()) for x in n_reservoir_range.split(',')]
            spectral_options = [float(x.strip()) for x in spectral_radius_range.split(',')]
        except ValueError:
            st.error("Invalid input. Please enter comma-separated numbers for the ranges.")
            st.stop()
        
    else:
        spectral_radius = st.slider("Spectral radius", 0.1, 2.0, 0.9, 0.05)
        n_reservoir = st.slider("Reservoir size", 10, 500, 50, 10)
    
    seed = st.number_input("Random seed", value=0)
    np.random.seed(seed)

    # -----------------------------
    # 3️⃣ Train ESN Button
    # -----------------------------
    if st.button("Train ESN Model"):
        st.session_state['train_data'] = {
            'X_train': train_df[input_cols_train].values,
            'y_train': train_df[output_cols_train].values,
            'input_cols': input_cols_train,
            'output_cols': output_cols_train
        }

        # Normalize
        X_train = st.session_state['train_data']['X_train']
        y_train = st.session_state['train_data']['y_train']

        input_mean = X_train.mean(axis=0)
        input_std  = X_train.std(axis=0)
        X_train_norm = (X_train - input_mean) / input_std
        
        output_mean = y_train.mean(axis=0)
        output_std  = y_train.std(axis=0)
        y_train_norm = (y_train - output_mean) / output_std
        
        # Transpose
        X_train_norm = X_train_norm.T
        y_train_norm = y_train_norm.T
        n_inputs = X_train_norm.shape[0]

        # -----------------------------
        # Function to train and evaluate ESN (now includes R2 score)
        # -----------------------------
        @st.cache_data
        def train_and_evaluate_esn(n_res, sr, input_norm, output_norm, n_in, output_std, output_mean):
            np.random.seed(seed)
            Win = (np.random.rand(n_res, n_in) * 2 - 1) * 0.1
            W = np.random.rand(n_res, n_res) * 2 - 1
            W = W * (sr / max(abs(eig(W)[0])))
            
            x = np.zeros((n_res, 1))
            X_res = []
            for t in range(input_norm.shape[1]):
                u = input_norm[:, t].reshape(-1, 1)
                x = np.tanh(Win @ u + W @ x)
                X_res.append(x)
            
            X_res = np.hstack(X_res)
            
            # Use pseudo-inverse for robust linear regression
            Wout = output_norm @ X_res.T @ np.linalg.pinv(X_res @ X_res.T)

            # Check R-squared on training data
            y_pred_norm_train = Wout @ X_res
            y_pred_train = y_pred_norm_train * output_std + output_mean
            
            # Flatten to handle multi-output correctly
            r2 = r2_score(y_train.flatten(), y_pred_train.T.flatten())
            
            return Win, W, Wout, r2

        # -----------------------------
        # Grid Search
        # -----------------------------
        if grid_search:
            st.info("Performing grid search... This may take time for large reservoirs.")
            best_r2 = -np.inf
            best_params = {}
            
            for n_res_try, sr_try in itertools.product(reservoir_options, spectral_options):
                Win_try, W_try, Wout_try, r2 = train_and_evaluate_esn(
                    n_res_try, sr_try, X_train_norm, y_train_norm, n_inputs, output_std, output_mean
                )
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_params = {
                        'n_reservoir': n_res_try,
                        'spectral_radius': sr_try,
                        'Win': Win_try,
                        'W': W_try,
                        'Wout': Wout_try
                    }
                    
            if best_r2 <= 0:
                st.warning(f"Could not find a positive R-squared score. Best R²: {best_r2:.4f}")
                st.info("Try adjusting the ranges for your grid search.")
            else:
                st.success(f"Best grid search R²: {best_r2:.4f} | n_reservoir: {best_params['n_reservoir']}, spectral_radius: {best_params['spectral_radius']}")
            
            Win, W, Wout = best_params['Win'], best_params['W'], best_params['Wout']
            n_reservoir = best_params['n_reservoir']
            
        else:
            Win, W, Wout, r2_train = train_and_evaluate_esn(
                n_reservoir, spectral_radius, X_train_norm, y_train_norm, n_inputs, output_std, output_mean
            )
            if r2_train <= 0:
                st.warning(f"The R² score is {r2_train:.4f}. This indicates a poor model fit.")
                st.info("You may want to try the grid search option to find better hyperparameters.")
            else:
                st.success(f"✅ ESN Model Trained Successfully! R² on training data: {r2_train:.4f}")

        # Store model in session
        st.session_state['trained_model'] = {
            'Win': Win,
            'W': W,
            'Wout': Wout,
            'n_reservoir': n_reservoir,
            'input_mean': input_mean,
            'input_std': input_std,
            'output_mean': output_mean,
            'output_std': output_std,
            'input_cols': input_cols_train,
            'output_cols': output_cols_train,
        }
        
# -----------------------------
# 4️⃣ Upload Test File
# -----------------------------
test_file = st.file_uploader("Upload Test Data (Excel or CSV)", type=["xlsx", "csv"], key="test_uploader")

if test_file and 'trained_model' in st.session_state:
    file_type_test = test_file.name.split('.')[-1]
    
    if file_type_test == 'csv':
        test_df = pd.read_csv(test_file)
    elif file_type_test == 'xlsx':
        excel_sheets_test = pd.ExcelFile(test_file).sheet_names
        selected_sheet_test = st.selectbox("Select the sheet for test data", excel_sheets_test)
        test_df = pd.read_excel(test_file, sheet_name=selected_sheet_test)

    st.subheader("Test Data Preview")
    st.dataframe(test_df.head())

    input_cols_test = st.multiselect(
        "Select input columns for testing", 
        test_df.columns.tolist(), 
        default=st.session_state['trained_model']['input_cols']
    )

    if st.button("Predict on Test File"):
        model = st.session_state['trained_model']
        
        # Check if selected input columns match training input columns
        if set(input_cols_test) != set(model['input_cols']):
            st.error("Selected test input columns must match the training input columns.")
            st.stop()

        X_test = test_df[input_cols_test].values
        X_test_norm = (X_test - model['input_mean']) / model['input_std']
        X_test_norm = X_test_norm.T

        # Predict
        x = np.zeros((model['n_reservoir'], 1))
        X_test_res = []
        for t in range(X_test_norm.shape[1]):
            u = X_test_norm[:, t].reshape(-1, 1)
            x = np.tanh(model['Win'] @ u + model['W'] @ x)
            X_test_res.append(x)
        
        X_test_res = np.hstack(X_test_res)
        y_pred_norm = (model['Wout'] @ X_test_res).T
        y_pred = y_pred_norm * model['output_std'] + model['output_mean']

        # -----------------------------
        # Show results
        # -----------------------------
        st.subheader("Predicted Output")
        pred_df = pd.DataFrame(y_pred, columns=model['output_cols'])
        st.dataframe(pred_df)

        # -----------------------------
        # Plot predicted vs actual if available
        # -----------------------------
        has_actual_output = any(col in test_df.columns for col in model['output_cols'])
        if has_actual_output:
            y_actual = test_df[model['output_cols']].values
            R2_test = r2_score(y_actual, y_pred)
            st.write(f"R² score on test data: {R2_test:.4f}")

            st.subheader("Predicted vs Actual")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(y_actual, 'b', label="Actual", linewidth=2)
            ax.plot(y_pred, 'r--', label="Predicted", linewidth=2)
            ax.set_xlabel("Sample")
            ax.set_ylabel("Output")
            ax.set_title("ESN Prediction on Test Set")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.subheader("Predicted Values")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(y_pred, 'r--', label="Predicted", linewidth=2)
            ax.set_xlabel("Sample")
            ax.set_ylabel("Output")
            ax.set_title("ESN Prediction")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        
        # Download predictions
        output = BytesIO()
        pred_df.to_excel(output, index=False, engine='openpyxl')
        st.download_button(
            label="Download Predictions as Excel",
            data=output.getvalue(),
            file_name="ESN_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
