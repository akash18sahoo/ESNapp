import streamlit as st
import pandas as pd
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from io import BytesIO
import itertools
from sklearn.model_selection import train_test_split

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="ESN Predictor", layout="wide")
st.title("🚀 Echo State Network Predictor (ESNapp)")

st.markdown("""
<style>
    .stButton>button {font-size:16px; padding:0.5em 1.2em;}
    .stDownloadButton>button {font-size:16px; padding:0.5em 1.2em;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 1️⃣ Upload Training Data
# -----------------------------
st.header("1️⃣ Upload & Configure Training Data")

file_uploader = st.file_uploader("Upload Training Data (Excel or CSV)", type=["xlsx", "csv"])

if file_uploader:
    file_type = file_uploader.name.split('.')[-1]
    if file_type == 'csv':
        train_df = pd.read_csv(file_uploader)
    else:
        sheets = pd.ExcelFile(file_uploader).sheet_names
        sheet = st.selectbox("Select Sheet for Training Data", sheets)
        train_df = pd.read_excel(file_uploader, sheet_name=sheet)

    st.subheader("Preview of Training Data")
    st.dataframe(train_df.head())

    input_cols_train = st.multiselect("Select **Input Columns**", train_df.columns.tolist(), default=train_df.columns[:-1].tolist())
    output_cols_train = st.multiselect("Select **Output Column(s)**", train_df.columns.tolist(), default=[train_df.columns[-1]])

    st.divider()
    st.header("2️⃣ ESN Hyperparameters")

    use_grid = st.checkbox("🔎 Use Grid Search for Hyperparameter Tuning", value=True)

    if use_grid:
        n_res_range = st.text_input("Reservoir sizes (comma separated)", "50,100,200,300,400")
        sr_range = st.text_input("Spectral radius values", "0.6,0.8,1.0,1.2")
        alpha_range = st.text_input("Regularization α values", "1e-6,1e-4,1e-2,1e-1")

        try:
            reservoir_options = [int(x.strip()) for x in n_res_range.split(',')]
            sr_options = [float(x.strip()) for x in sr_range.split(',')]
            alpha_options = [float(x.strip()) for x in alpha_range.split(',')]
        except ValueError:
            st.error("⚠️ Please enter comma-separated numeric values.")
            st.stop()
    else:
        spectral_radius = st.slider("Spectral Radius", 0.1, 2.0, 0.9, 0.05)
        n_reservoir = st.slider("Reservoir Size", 10, 1000, 200, 10)
        alpha = st.number_input("Regularization α", value=1e-4, format="%.1e")

    seed = st.number_input("Random Seed", value=42)
    np.random.seed(seed)

    # -----------------------------
    # Normalize
    # -----------------------------
    X = train_df[input_cols_train].values
    y = train_df[output_cols_train].values

    input_mean, input_std = X.mean(axis=0), X.std(axis=0) + 1e-12
    output_mean, output_std = y.mean(axis=0), y.std(axis=0) + 1e-12

    X_norm = ((X - input_mean) / input_std).T
    y_norm = ((y - output_mean) / output_std).T
    n_inputs = X_norm.shape[0]

    # -----------------------------
    # ESN training function
    # -----------------------------
    def train_esn(n_res, sr, alpha, Xtr, Ytr):
        np.random.seed(seed)
        Win = (np.random.rand(n_res, n_inputs) * 2 - 1) * 0.1
        W = np.random.rand(n_res, n_res) * 2 - 1
        W *= sr / max(abs(eig(W)[0]))

        x = np.zeros((n_res, 1))
        Xres = []
        for t in range(Xtr.shape[1]):
            u = Xtr[:, t].reshape(-1, 1)
            x = np.tanh(Win @ u + W @ x)
            Xres.append(x)
        Xres = np.hstack(Xres)

        # Ridge regression for output weights
        Wout = Ytr @ Xres.T @ np.linalg.inv(Xres @ Xres.T + alpha * np.eye(Xres.shape[0]))

        # Prediction on training
        Ypred_tr = Wout @ Xres
        r2_tr = r2_score(Ytr.T.flatten(), Ypred_tr.T.flatten())
        return Win, W, Wout, r2_tr

    # -----------------------------
    # Train ESN
    # -----------------------------
    if st.button("🚀 Train ESN Model"):
        if use_grid:
            st.info("🔎 Running Grid Search... (best chosen on validation R²)")
            X_train, X_val, y_train, y_val = train_test_split(X_norm.T, y_norm.T, test_size=0.2, random_state=seed)
            X_train, X_val = X_train.T, X_val.T
            y_train, y_val = y_train.T, y_val.T

            best_r2_val = -np.inf
            progress = st.progress(0)
            total = len(reservoir_options) * len(sr_options) * len(alpha_options)
            done = 0

            for nr, sr, al in itertools.product(reservoir_options, sr_options, alpha_options):
                Win_, W_, Wout_, _ = train_esn(nr, sr, al, X_train, y_train)

                # Evaluate on validation
                x = np.zeros((nr, 1))
                Xres_val = []
                for t in range(X_val.shape[1]):
                    u = X_val[:, t].reshape(-1, 1)
                    x = np.tanh(Win_ @ u + W_ @ x)
                    Xres_val.append(x)
                Xres_val = np.hstack(Xres_val)
                ypred_val = Wout_ @ Xres_val
                r2_val = r2_score(y_val.T.flatten(), ypred_val.T.flatten())

                if r2_val > best_r2_val:
                    best_r2_val = r2_val
                    best_params = dict(nr=nr, sr=sr, al=al, Win=Win_, W=W_, Wout=Wout_)

                done += 1
                progress.progress(done / total)

            if best_r2_val > 0:
                st.success(f"✅ Best Validation R² = {best_r2_val:.4f} | Reservoir={best_params['nr']} | SR={best_params['sr']} | α={best_params['al']}")
            else:
                st.warning(f"⚠️ Validation R² is low ({best_r2_val:.4f}). Consider expanding the search ranges.")

            Win, W, Wout = best_params['Win'], best_params['W'], best_params['Wout']
            n_reservoir, spectral_radius, alpha = best_params['nr'], best_params['sr'], best_params['al']

        else:
            Win, W, Wout, r2_train = train_esn(n_reservoir, spectral_radius, alpha, X_norm, y_norm)
            st.info(f"Training R² = {r2_train:.4f}")

        # Save model
        st.session_state['model'] = dict(
            Win=Win, W=W, Wout=Wout,
            n_reservoir=n_reservoir,
            input_mean=input_mean, input_std=input_std,
            output_mean=output_mean, output_std=output_std,
            input_cols=input_cols_train, output_cols=output_cols_train
        )

        st.success("🎉 Model Trained and Ready for Testing!")

# -----------------------------
# 3️⃣ Testing
# -----------------------------
st.divider()
st.header("3️⃣ Test the Model")

test_file = st.file_uploader("Upload Test Data (Excel or CSV)", type=["xlsx", "csv"], key="testfile")

if test_file and 'model' in st.session_state:
    if test_file.name.endswith('csv'):
        test_df = pd.read_csv(test_file)
    else:
        sheets = pd.ExcelFile(test_file).sheet_names
        sheet = st.selectbox("Select Sheet for Test Data", sheets)
        test_df = pd.read_excel(test_file, sheet_name=sheet)

    st.subheader("Preview of Test Data")
    st.dataframe(test_df.head())

    input_cols_test = st.multiselect("Select Test Input Columns", test_df.columns.tolist(),
                                     default=st.session_state['model']['input_cols'])

    if st.button("📈 Predict on Test Data"):
        mdl = st.session_state['model']

        # Check inputs
        if set(input_cols_test) != set(mdl['input_cols']):
            st.error("⚠️ Test input columns must match training input columns.")
            st.stop()

        X_test = test_df[input_cols_test].values
        X_test_norm = ((X_test - mdl['input_mean']) / mdl['input_std']).T

        x = np.zeros((mdl['n_reservoir'], 1))
        Xres_test = []
        for t in range(X_test_norm.shape[1]):
            u = X_test_norm[:, t].reshape(-1, 1)
            x = np.tanh(mdl['Win'] @ u + mdl['W'] @ x)
            Xres_test.append(x)
        Xres_test = np.hstack(Xres_test)

        ypred_norm = (mdl['Wout'] @ Xres_test).T
        ypred = ypred_norm * mdl['output_std'] + mdl['output_mean']

        pred_df = pd.DataFrame(ypred, columns=mdl['output_cols'])
        st.subheader("Predicted Output")
        st.dataframe(pred_df)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        if set(mdl['output_cols']).issubset(test_df.columns):
            y_actual = test_df[mdl['output_cols']].values
            r2_test = r2_score(y_actual, ypred)
            st.success(f"✅ R² on Test Data = {r2_test:.4f}")

            ax.plot(y_actual, 'b', label='Actual', linewidth=2)
            ax.plot(ypred, 'r--', label='Predicted', linewidth=2)
            ax.set_title("Predicted vs Actual")
        else:
            ax.plot(ypred, 'r--', label='Predicted', linewidth=2)
            ax.set_title("Predicted Output")

        ax.set_xlabel("Sample")
        ax.set_ylabel("Output")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Download
        out = BytesIO()
        pred_df.to_excel(out, index=False, engine='openpyxl')
        st.download_button("⬇️ Download Predictions as Excel",
                           data=out.getvalue(),
                           file_name="ESN_predictions.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
