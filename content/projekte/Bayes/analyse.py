from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import pymc as pm
import arviz as az


# ------------------------------------------------------------
# 1. DATEN LADEN / GENERIEREN
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

    # Wahre Sterbewahrscheinlichkeiten
    # Beta(8, 22) -> Mittelwert ca. 26.7 %
    p_true = rng.beta(8, 22, size=20)

    # Todesfälle binomial simulieren
    deaths = rng.binomial(n_patients, p_true)

    # Beobachtete Sterberate
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
# 2. BAYES BERECHNUNG - NO POOLING
# ------------------------------------------------------------
def berechne_bayes(df_obs, df_truth, n_0, prior_mean=0.267):
    """
    Bayesianische Schätzung mit demselben festen Prior
    für jedes Krankenhaus.
    """

    alpha_prior = max(0.1, n_0 * prior_mean)
    beta_prior = max(0.1, n_0 * (1 - prior_mean))

    alpha_post = alpha_prior + df_obs["deaths"]

    beta_post = (
        beta_prior
        + (df_obs["n_patients"] - df_obs["deaths"])
    )

    df_res = df_obs.copy()

    # Posterior-Mittelwert
    df_res["p_bayes"] = (
        alpha_post / (alpha_post + beta_post)
    )

    # Fehler
    mse_naiv = np.mean(
        (df_res["p_observed"] - df_truth["p_true"]) ** 2
    )

    mse_bayes = np.mean(
        (df_res["p_bayes"] - df_truth["p_true"]) ** 2
    )

    return df_res, mse_naiv, mse_bayes


# ------------------------------------------------------------
# 3. PARTIAL POOLING - HIERARCHISCHES BAYES-MODELL
# ------------------------------------------------------------
@st.cache_resource
def berechne_partial_pooling(df_obs):
    """
    Hierarchisches Bayesianisches Modell.

    Jedes Krankenhaus bekommt eine eigene Sterbewahrscheinlichkeit p_j.
    Die Krankenhäuser teilen sich aber eine gemeinsame Population.
    """

    n_patients = df_obs["n_patients"].to_numpy()
    deaths = df_obs["deaths"].to_numpy()

    n_hospitals = len(df_obs)

    with pm.Model() as model:

        # ----------------------------------------------------
        # Gemeinsames Zentrum der Krankenhaus-Population
        # ----------------------------------------------------
        mu = pm.Normal(
            "mu",
            mu=-1,
            sigma=1
        )

        # ----------------------------------------------------
        # Streuung zwischen den Krankenhäusern
        # ----------------------------------------------------
        tau = pm.HalfNormal(
            "tau",
            sigma=1
        )

        # ----------------------------------------------------
        # Krankenhaus-spezifische Effekte
        #
        # z_j = logit(p_j)
        #
        # Alle Krankenhäuser kommen aus derselben
        # Normalverteilung mit Zentrum mu und Streuung tau.
        # ----------------------------------------------------
        z = pm.Normal(
            "z",
            mu=mu,
            sigma=tau,
            shape=n_hospitals
        )

        # ----------------------------------------------------
        # Zurück auf Wahrscheinlichkeitsskala
        #
        # p_j = sigmoid(z_j)
        # ----------------------------------------------------
        p = pm.Deterministic(
            "p",
            pm.math.sigmoid(z)
        )

        # ----------------------------------------------------
        # Beobachtungsmodell
        #
        # y_j ~ Binomial(n_j, p_j)
        # ----------------------------------------------------
        y = pm.Binomial(
            "y",
            n=n_patients,
            p=p,
            observed=deaths
        )

        # ----------------------------------------------------
        # MCMC
        # ----------------------------------------------------
        trace = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            random_seed=42,
            target_accept=0.9,
            progressbar=True
        )

    # --------------------------------------------------------
    # Posterior-Mittelwert für jedes Krankenhaus
    # --------------------------------------------------------
    p_posterior_mean = (
        trace.posterior["p"]
        .mean(dim=("chain", "draw"))
        .values
    )

    # --------------------------------------------------------
    # 95%-Credible-Intervals
    # --------------------------------------------------------
    p_hdi = az.hdi(
        trace.posterior["p"],
        prob=0.95
    )

    p_lower = p_hdi.sel(ci_bound="lower").values
    p_upper = p_hdi.sel(ci_bound="upper").values

    # --------------------------------------------------------
    # Ergebnisse an ursprünglichen DataFrame anhängen
    # --------------------------------------------------------
    df_res = df_obs.copy()

    df_res["p_partial"] = p_posterior_mean
    df_res["p_partial_lower"] = p_lower
    df_res["p_partial_upper"] = p_upper

    return df_res, trace


# ------------------------------------------------------------
# 4. GESAMTVERGLEICH
# ------------------------------------------------------------
def vergleiche_modelle(df_obs, df_truth, n_0):
    """
    Führt No Pooling und Partial Pooling durch
    und berechnet die MSEs.
    """

    # No Pooling
    df_bayes, mse_naiv, mse_bayes = berechne_bayes(
        df_obs,
        df_truth,
        n_0
    )

    # Partial Pooling
    df_partial, trace = berechne_partial_pooling(
        df_obs
    )

    # Partial-Pooling-Schätzung hinzufügen
    df_bayes["p_partial"] = df_partial["p_partial"]

    # Credible Intervals
    df_bayes["p_partial_lower"] = (
        df_partial["p_partial_lower"]
    )

    df_bayes["p_partial_upper"] = (
        df_partial["p_partial_upper"]
    )

    # MSE Partial Pooling
    mse_partial = np.mean(
        (
            df_bayes["p_partial"]
            - df_truth["p_true"]
        ) ** 2
    )

    return (
        df_bayes,
        trace,
        mse_naiv,
        mse_bayes,
        mse_partial
    )


# ------------------------------------------------------------
# 5. VISUALISIERUNG
# ------------------------------------------------------------
def erstelle_plot(df_res, df_truth, n_0):
    """
    Vergleich:
    - Ground Truth
    - beobachtete Rate
    - No Pooling
    - Partial Pooling
    """

    fig = go.Figure()

    # --------------------------------------------------------
    # Ground Truth
    # --------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=df_res["hospital"],
            y=df_truth["p_true"],
            mode="markers",
            name="Ground Truth (Wahrheit)",
            marker=dict(
                symbol="star",
                size=11,
                color="green"
            ),
        )
    )

    # --------------------------------------------------------
    # Naive Rohdaten
    # --------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=df_res["hospital"],
            y=df_res["p_observed"],
            mode="markers",
            name="Beobachtet (Naiv)",
            marker=dict(
                size=7,
                color="gray",
                opacity=0.6
            ),
        )
    )

    # --------------------------------------------------------
    # Bayesianisches No Pooling
    # --------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=df_res["hospital"],
            y=df_res["p_bayes"],
            mode="markers+lines",
            name=f"Bayes No Pooling (N₀={n_0})",
            marker=dict(
                size=9,
                color="#1f77b4"
            ),
        )
    )

    # --------------------------------------------------------
    # Partial Pooling
    # --------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=df_res["hospital"],
            y=df_res["p_partial"],
            mode="markers+lines",
            name="Partial Pooling",
            marker=dict(
                size=9,
                color="red"
            ),
            error_y=dict(
                type="data",
                symmetric=False,
                array=(
                    df_res["p_partial_upper"]
                    - df_res["p_partial"]
                ),
                arrayminus=(
                    df_res["p_partial"]
                    - df_res["p_partial_lower"]
                ),
                thickness=1.5,
                width=4
            )
        )
    )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------
    fig.update_layout(
        title="Vergleich: No Pooling vs. Partial Pooling",
        xaxis_title=(
            "Krankenhaus "
            "(Sortiert nach steigender Fallzahl n)"
        ),
        yaxis_title="Geschätzte Sterberate",
        yaxis=dict(range=[-0.02, 0.6]),
        legend=dict(
            orientation="h",
            y=1.12,
            x=0
        ),
        margin=dict(
            l=20,
            r=20,
            t=70,
            b=20
        ),
    )

    return fig