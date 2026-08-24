import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Projektordner in den Python-Pfad einfügen
ORDNER = Path(__file__).parent

if str(ORDNER) not in sys.path:
    sys.path.insert(0, str(ORDNER))

# Eigene Logik importieren
import analyse  # noqa: E402


# ============================================================
# DATEN LADEN
# ============================================================

df_obs, df_truth = analyse.lade_daten()


# ============================================================
# TITEL
# ============================================================

st.title("🏥 Retten kleine Krankenhäuser weniger Leben?")

st.caption(
    "Sommerakademie Künstliche Intelligenz | "
    "Thema: Bayesian Methods"
)


# ============================================================
# TABS
# ============================================================

(
    tab_frage,
    tab_daten,
    tab_naiv,
    tab_ident,
    tab_ergebnis,
    tab_grenzen,
) = st.tabs(
    [
        "1 · Frage",
        "2 · Daten & EDA",
        "3 · Naive Analyse",
        "4 · Bayesianisches Modell",
        "5 · Ergebnis",
        "6 · Limitationen",
    ]
)


# ===================================================================
# 1 · FRAGE
# ===================================================================

with tab_frage:

    st.markdown("## Die Frage")

    st.markdown(
        """
        Kleine Krankenhäuser sichern die medizinische Versorgung vor Ort,
        aber ihre Fallzahlen sind oft klein.

        Wer im Jahr nur eine Handvoll Eingriffe durchführt, unterliegt bei
        Komplikationen extremen Zufallsschwankungen.

        Gesundheitspolitisch und statistisch ist die Auswertung dieser
        Häuser deshalb besonders schwierig.

        > **Ist das Behandlungsrisiko in kleinen Krankenhäusern wirklich höher?**

        Der naheliegende Weg wäre, die beobachteten Sterberaten direkt
        miteinander zu vergleichen.

        Das kann jedoch irreführend sein: Kleine Fallzahlen erzeugen durch
        reines Zufallsrauschen extreme Ausschläge.

        Ein Krankenhaus mit nur 5 Patienten kann beispielsweise leicht eine
        beobachtete Sterberate von 0 %, 20 %, 40 % oder sogar 60 % haben,
        obwohl seine tatsächliche Qualität völlig normal ist.
        """
    )

    st.markdown("## Die Simulation")

    st.markdown(
        """
        Wir beantworten die Frage zunächst über eine kontrollierte
        **Datensimulation**.

        Wir erzeugen ein Gesundheitssystem mit **20 Krankenhäusern**, deren
        Fallzahlen bewusst stark variieren:

        - **Sehr kleine Häuser:** 5 bis 20 Patienten
        - **Mittlere Häuser:** 30 bis 150 Patienten
        - **Große Häuser:** 200 bis 1.000 Patienten

        Für jedes Krankenhaus wird eine **wahre Sterbewahrscheinlichkeit**
        simuliert.

        Aus dieser wahren Sterbewahrscheinlichkeit werden anschließend die
        tatsächlich beobachteten Todesfälle simuliert.
        """
    )

    st.markdown(
        """
        **Warum die Simulation hilfreich ist:**

        In einer realen Studie kennen wir die tatsächliche Sterbewahrscheinlichkeit
        eines Krankenhauses nicht.

        In unserer Simulation kennen wir sie jedoch. Dadurch können wir
        verschiedene statistische Methoden direkt mit der Wahrheit vergleichen.
        """
    )


# ===================================================================
# 2 · DATEN & EDA
# ===================================================================

with tab_daten:

    st.markdown("## Explorative Datenanalyse (EDA)")

    st.markdown(
        """
        Unsere simulierten Daten enthalten:

        - **`n_patients`**: Anzahl der Patienten
        - **`deaths`**: beobachtete Todesfälle
        - **`p_observed`**: naive Sterberate
        - **`p_true`**: tatsächliche Sterbewahrscheinlichkeit
        """
    )

    st.markdown("### Beobachtete Daten")

    st.dataframe(
        df_obs,
        use_container_width=True
    )

    st.markdown("### Fallzahlen der Krankenhäuser")

    st.bar_chart(
        df_obs.set_index("hospital")["n_patients"]
    )

    st.markdown("### Beobachtete Sterberaten")

    fig_eda = go.Figure()

    fig_eda.add_trace(
        go.Scatter(
            x=df_obs["hospital"],
            y=df_obs["p_observed"],
            mode="markers",
            name="Beobachtete Sterberate",
            marker=dict(size=9),
        )
    )

    fig_eda.add_trace(
        go.Scatter(
            x=df_truth["hospital"],
            y=df_truth["p_true"],
            mode="markers",
            name="Ground Truth",
            marker=dict(
                symbol="star",
                size=11,
            ),
        )
    )

    fig_eda.update_layout(
        title="Beobachtete Sterberate vs. Ground Truth",
        xaxis_title="Krankenhaus",
        yaxis_title="Sterbewahrscheinlichkeit",
        yaxis=dict(range=[-0.02, 0.6]),
    )

    st.plotly_chart(
        fig_eda,
        use_container_width=True
    )


# ===================================================================
# 3 · NAIVE ANALYSE
# ===================================================================

with tab_naiv:

    st.markdown("## Naive Analyse (No Pooling)")

    st.markdown(
        r"""
        Die einfachste Schätzung ist:

        $$\hat p_j = \frac{y_j}{n_j}$$

        Wir verwenden also für jedes Krankenhaus einfach die beobachtete
        Sterberate.

        Das Problem:

        **Kleine Fallzahlen erzeugen sehr viel Zufallsrauschen.**
        """
    )

    fig_naiv = go.Figure()

    # Ground Truth
    fig_naiv.add_trace(
        go.Scatter(
            x=df_obs["hospital"],
            y=df_truth["p_true"],
            mode="markers",
            name="Ground Truth",
            marker=dict(
                symbol="star",
                size=11,
            ),
        )
    )

    # Beobachtete Werte
    fig_naiv.add_trace(
        go.Scatter(
            x=df_obs["hospital"],
            y=df_obs["p_observed"],
            mode="markers",
            name="Beobachtet",
            marker=dict(size=8),
        )
    )

    fig_naiv.update_layout(
        title="Naive Beobachtung vs. Wahrheit",
        xaxis_title="Krankenhaus (steigende Fallzahl)",
        yaxis_title="Sterberate",
        yaxis=dict(range=[-0.02, 0.6]),
    )

    st.plotly_chart(
        fig_naiv,
        use_container_width=True
    )

    # MSE der naiven Methode
    mse_naiv = np.mean(
        (
            df_obs["p_observed"]
            - df_truth["p_true"]
        ) ** 2
    )

    st.metric(
        "MSE der naiven Schätzung",
        f"{mse_naiv:.5f}"
    )


# ===================================================================
# 4 · BAYESIANISCHES MODELL
# ===================================================================

with tab_ident:

    st.markdown("## Bayesianisches Modell")

    st.markdown(
        r"""
        ### 1. Likelihood

        Für jedes Krankenhaus gilt:

        $$y_j \sim \operatorname{Binomial}(n_j,p_j)$$

        Dabei sind:

        - $n_j$: Anzahl der Patienten
        - $y_j$: Anzahl der Todesfälle
        - $p_j$: wahre Sterbewahrscheinlichkeit

        ---

        ### 2. Einfaches Bayesianisches Modell

        Zunächst können wir für jedes Krankenhaus denselben Prior verwenden:

        $$p_j \sim \operatorname{Beta}(\alpha,\beta)$$

        Daraus entsteht ein eigener Posterior für jedes Krankenhaus.

        Das ist **No Pooling mit gemeinsamem Prior**.

        ---

        ### 3. Partial Pooling

        Beim Partial Pooling modellieren wir die Krankenhäuser gemeinsam:

        $$\operatorname{logit}(p_j) \sim N(\mu,\tau^2)$$

        Dabei beschreibt:

        - $\mu$: typisches Niveau der Krankenhäuser
        - $\tau$: Unterschiede zwischen den Krankenhäusern

        Das Modell lernt $\mu$ und $\tau$ aus den Daten.

        Dadurch können die Krankenhäuser voneinander lernen:

        **Kleine Krankenhäuser werden stärker zur gemeinsamen Mitte gezogen,
        große Krankenhäuser stärker durch ihre eigenen Daten bestimmt.**
        """
    )


# ===================================================================
# 5 · ERGEBNIS
# ===================================================================

with tab_ergebnis:

    st.markdown(
        "## No Pooling vs. Partial Pooling"
    )

    st.markdown(
        """
        Hier vergleichen wir drei Dinge:

        1. **Naive Schätzung:** Nur die beobachtete Rate
        2. **Bayesianisches Shrinkage:** Fester Prior
        3. **Partial Pooling:** Gemeinsames hierarchisches Modell
        """
    )

    # ------------------------------------------------------------
    # SLIDER
    # ------------------------------------------------------------

    st.markdown("### Einfaches Bayesianisches Shrinkage")

    n_0 = st.slider(
        "Stärke des festen Priors ($N_0$)",
        min_value=0,
        max_value=100,
        value=20,
        step=5,
        help=(
            "N₀ entspricht ungefähr der Stärke des Vorwissens "
            "in Form einer Pseudofallzahl."
        ),
    )

    st.markdown(
        r"""
        Bei $N_0=0$ entspricht das Ergebnis der naiven Schätzung.

        Je größer $N_0$ ist, desto stärker werden kleine Krankenhäuser
        zum Prior-Mittelwert gezogen.
        """
    )

    # ------------------------------------------------------------
    # NO POOLING
    # ------------------------------------------------------------

    df_res, mse_naiv, mse_bayes = analyse.berechne_bayes(
        df_obs,
        df_truth,
        n_0
    )

    # ------------------------------------------------------------
    # PARTIAL POOLING
    # ------------------------------------------------------------

    with st.spinner(
        "Partial-Pooling-Modell wird berechnet ..."
    ):

        df_partial, trace = (
            analyse.berechne_partial_pooling(
                df_obs
            )
        )

    # Ergebnisse zusammenführen
    df_res["p_partial"] = df_partial["p_partial"]

    df_res["p_partial_lower"] = (
        df_partial["p_partial_lower"]
    )

    df_res["p_partial_upper"] = (
        df_partial["p_partial_upper"]
    )

    # ------------------------------------------------------------
    # MSE PARTIAL POOLING
    # ------------------------------------------------------------

    mse_partial = np.mean(
        (
            df_res["p_partial"]
            - df_truth["p_true"]
        ) ** 2
    )

    # ------------------------------------------------------------
    # HYPERPARAMETER
    # ------------------------------------------------------------

    mu_mean = (
        trace.posterior["mu"]
        .mean()
        .item()
    )

    tau_mean = (
        trace.posterior["tau"]
        .mean()
        .item()
    )

    # ------------------------------------------------------------
    # KENNZAHLEN
    # ------------------------------------------------------------

    st.markdown("### Modellvergleich")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "MSE Naiv",
        f"{mse_naiv:.5f}"
    )

    col2.metric(
        "MSE fester Prior",
        f"{mse_bayes:.5f}"
    )

    col3.metric(
        "MSE Partial Pooling",
        f"{mse_partial:.5f}"
    )

    # ------------------------------------------------------------
    # GELERNTE POPULATION
    # ------------------------------------------------------------

    st.markdown(
        "### Was lernt das Partial-Pooling-Modell?"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        r"$\mu$ – typisches Niveau",
        f"{mu_mean:.2f}"
    )

    col2.metric(
        r"$\tau$ – Unterschiede zwischen Krankenhäusern",
        f"{tau_mean:.2f}"
    )

    st.caption(
        "μ und τ liegen in diesem Modell auf der Logit-Skala."
    )

    # ------------------------------------------------------------
    # HAUPTPLOT
    # ------------------------------------------------------------

    st.markdown("### Vergleich der Schätzungen")

    fig = go.Figure()

    # Ground Truth
    fig.add_trace(
        go.Scatter(
            x=df_res["hospital"],
            y=df_truth["p_true"],
            mode="markers",
            name="Ground Truth",
            marker=dict(
                symbol="star",
                size=12,
            ),
        )
    )

    # Naive Beobachtung
    fig.add_trace(
        go.Scatter(
            x=df_res["hospital"],
            y=df_res["p_observed"],
            mode="markers",
            name="Beobachtet",
            marker=dict(
                size=8,
                opacity=0.6,
            ),
        )
    )

    # Fester Prior
    fig.add_trace(
        go.Scatter(
            x=df_res["hospital"],
            y=df_res["p_bayes"],
            mode="markers+lines",
            name=f"Fester Prior ($N_0={n_0}$)",
            marker=dict(size=8),
        )
    )

    # Partial Pooling
    fig.add_trace(
        go.Scatter(
            x=df_res["hospital"],
            y=df_res["p_partial"],
            mode="markers+lines",
            name="Partial Pooling",
            marker=dict(size=9),
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
                width=4,
            ),
        )
    )

    fig.update_layout(
        title=(
            "No Pooling vs. Bayesianisches Shrinkage "
            "vs. Partial Pooling"
        ),
        xaxis_title=(
            "Krankenhaus "
            "(sortiert nach steigender Fallzahl)"
        ),
        yaxis_title="Sterbewahrscheinlichkeit",
        yaxis=dict(
            range=[-0.02, 0.6]
        ),
        legend=dict(
            orientation="h",
            y=1.12,
            x=0,
        ),
        margin=dict(
            l=20,
            r=20,
            t=80,
            b=20,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------------------------------------------
    # ERGEBNISTABELLE
    # ------------------------------------------------------------

    st.markdown(
        "### Ergebnisse für jedes Krankenhaus"
    )

    tabelle = df_res[
        [
            "hospital",
            "n_patients",
            "deaths",
            "p_observed",
            "p_bayes",
            "p_partial",
            "p_partial_lower",
            "p_partial_upper",
        ]
    ].copy()

    tabelle.columns = [
        "Krankenhaus",
        "Patienten",
        "Todesfälle",
        "Beobachtet",
        "Fester Prior",
        "Partial Pooling",
        "PP 95%-Untergrenze",
        "PP 95%-Obergrenze",
    ]

    st.dataframe(
        tabelle,
        use_container_width=True,
    )

    # ------------------------------------------------------------
    # ERKLÄRUNG
    # ------------------------------------------------------------

    st.markdown("### Was sehen wir?")

    st.markdown(
        """
        Bei Krankenhäusern mit wenigen Patienten ist die beobachtete
        Sterberate sehr unsicher.

        Das Partial-Pooling-Modell nutzt zusätzlich die Information
        der anderen Krankenhäuser. Dadurch werden extreme Schätzungen
        kleiner Krankenhäuser stärker zur gemeinsamen Mitte gezogen.

        Bei großen Krankenhäusern sind die eigenen Daten sehr informativ.
        Deshalb bleibt ihre Partial-Pooling-Schätzung näher an der
        beobachteten Rate.

        **Das unterschiedliche Ausmaß dieses Shrinkage ist der zentrale
        Effekt des Partial Poolings.**
        """
    )


# ===================================================================
# 6 · LIMITATIONEN
# ===================================================================

with tab_grenzen:

    st.markdown("## Limitationen des Modells")

    st.markdown(
        r"""
        ### 1. Kein Risk Adjustment

        Das Modell berücksichtigt nicht, ob verschiedene Krankenhäuser
        unterschiedlich schwere Fälle behandeln.

        Ein Krankenhaus mit vielen schwerkranken Patienten könnte deshalb
        eine höhere Sterblichkeit haben, ohne schlechter zu behandeln.

        ---

        ### 2. Simulation statt realer Daten

        Unsere Ground Truth kennen wir nur, weil wir die Daten selbst
        simuliert haben.

        In einer realen Anwendung wäre die tatsächliche
        Sterbewahrscheinlichkeit unbekannt.

        ---

        ### 3. Modellannahmen

        Beim Partial Pooling nehmen wir an:

        $$\operatorname{logit}(p_j) \sim N(\mu,\tau^2)$$

        Ob diese Annahme angemessen ist, müsste bei realen Daten überprüft
        werden.

        ---

        ### 4. Prior für $\mu$ und $\tau$

        Auch $\mu$ und $\tau$ erhalten Priors.

        Deshalb sollte man untersuchen, wie empfindlich die Ergebnisse
        gegenüber unterschiedlichen Priors sind.

        ---

        ### 5. Over-Shrinkage

        Partial Pooling kann extreme Schätzungen kleiner Krankenhäuser
        reduzieren.

        Das ist normalerweise hilfreich, kann aber auch ein tatsächlich
        außergewöhnlich gutes oder schlechtes Krankenhaus zu stark zur
        gemeinsamen Mitte ziehen.
        """
    )