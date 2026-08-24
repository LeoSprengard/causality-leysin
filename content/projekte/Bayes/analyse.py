from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ------------------------------------------------------------
# 1. DATEN LADEN / GENERIEREN (Gecached für Streamlit)
# ------------------------------------------------------------
@st.cache_data
def lade_daten():
    """Generiert die Daten und gibt sie als DataFrames zurück."""
    rng = np.random.default_rng(42)

    n_patients = np.array([
        5, 8, 12, 15, 20,
        30, 40, 50, 70, 90,
        120, 150, 200, 250, 300,
        400, 500, 600, 800, 1000
    ])
    hospital = [f"H{i:02d}" for i in range(1, 21)]

    # Wahre Sterbewahrscheinlichkeiten (Beta(8, 22) -> Mean ~ 26.7%)
    p_true = rng.beta(8, 22, size=20)

    # Todesfälle binomial simulieren
    deaths = rng.binomial(n_patients, p_true)
    p_observed = deaths / n_patients

    df_obs = pd.DataFrame({
        "hospital": hospital,
        "n_patients": n_patients,
        "deaths": deaths,
        "p_observed": p_observed
    })

    df_truth = pd.DataFrame({
        "hospital": hospital,
        "p_true": p_true
    })

    return df_obs, df_truth


# ------------------------------------------------------------
# 2. BAYES BERECHNUNG
# ------------------------------------------------------------
def berechne_bayes(df_obs, df_truth, n_0, prior_mean=0.267):
    """Berechnet die Bayes-Posterior-Schätzung und die Fehler (MSE)."""
    alpha_prior = max(0.1, n_0 * prior_mean)
    beta_prior = max(0.1, n_0 * (1 - prior_mean))

    alpha_post = alpha_prior + df_obs["deaths"]
    beta_post = beta_prior + (df_obs["n_patients"] - df_obs["deaths"])

    df_res = df_obs.copy()
    df_res["p_bayes"] = alpha_post / (alpha_post + beta_post)

    mse_naiv = np.mean((df_res["p_observed"] - df_truth["p_true"]) ** 2)
    mse_bayes = np.mean((df_res["p_bayes"] - df_truth["p_true"]) ** 2)

    return df_res, mse_naiv, mse_bayes


# ------------------------------------------------------------
# 3. VISUALISIERUNG
# ------------------------------------------------------------
def erstelle_plot(df_res, df_truth, n_0):
    """Erstellt das interaktive Plotly-Diagramm."""
    fig = go.Figure()

    # Ground Truth
    fig.add_trace(
        go.Scatter(
            x=df_res["hospital"],
            y=df_truth["p_true"],
            mode="markers",
            name="Ground Truth (Wahrheit)",
            marker=dict(symbol="star", size=11, color="green"),
        )
    )

    # Naive Rohdaten
    fig.add_trace(
        go.Scatter(
            x=df_res["hospital"],
            y=df_res["p_observed"],
            mode="markers",
            name="Beobachtet (Naiv)",
            marker=dict(size=7, color="gray", opacity=0.6),
        )
    )

    # Bayes-Schätzung
    fig.add_trace(
        go.Scatter(
            x=df_res["hospital"],
            y=df_res["p_bayes"],
            mode="markers+lines",
            name=f"Bayes-Schätzung (N₀={n_0})",
            marker=dict(size=9, color="#1f77b4"),
        )
    )

    fig.update_layout(
        title="Effekt des Priors auf kleine vs. große Krankenhäuser",
        xaxis_title="Krankenhaus (Sortiert nach steigender Fallzahl n)",
        yaxis_title="Geschätzte Sterberate",
        yaxis=dict(range=[-0.02, 0.6]),
        legend=dict(orientation="h", y=1.12, x=0),
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


