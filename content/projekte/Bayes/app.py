import sys
from pathlib import Path
import plotly.graph_objects as go
import streamlit as st

# Projektordner in den Python-Pfad einfügen
ORDNER = Path(__file__).parent
if str(ORDNER) not in sys.path:
    sys.path.insert(0, str(ORDNER))

# Eigene Logik & Styling importieren
import analyse  # noqa: E402
from utils.theming import FARBEN, merkkasten  # noqa: E402

# ------------------------------------------------------------
# Daten aus analyse.py laden
# ------------------------------------------------------------
df_obs, df_truth = analyse.lade_daten()

# 1. Überschrift & Einleitung
# ------------------------------------------------------------
st.title("🏥 Retten kleine Krankenhäuser weniger Leben?")
st.caption("Sommerakademie Künstliche Intelligenz | Thema: Bayesian Methods")


tab_frage, tab_daten, tab_naiv, tab_ident, tab_ergebnis, tab_grenzen = st.tabs(
    [
        "1 · Frage",
        "2 · Daten & EDA",
        "3 · Naive Analyse",
        "4 · Identifikation",
        "5 · Ergebnis",
        "6 · Limitationen",
    ]
)

# =================================================================== 1 Frage
with tab_frage:
    st.markdown("## Die Frage")
    st.markdown(
        """
        Kleine Krankenhäuser sichern die medizinische Versorgung vor Ort, aber ihre Fallzahlen sind winzig: 
        Wer im Jahr nur eine Handvoll Eingriffe durchführt, unterliegt bei Komplikationen extremen Zufallsschwankungen. 
        Gesundheitspolitisch und statistisch ist die Auswertung dieser Häuser deshalb eine der fehleranfälligsten Stellschrauben überhaupt. 

        > **Ist das Behandlungsrisiko in kleinen Krankenhäusern wirklich höher?**

        Der naheliegende Weg wäre, die beobachteten Sterberaten direkt zu vergleichen. Der taugt aber nichts: 
        Kleine Fallzahlen erzeugen durch reines Zufallsrauschen extreme Ausschläge (0 % oder 50 % Sterberate bei wenigen Patienten). 
        Wer kleine mit großen Kliniken eins zu eins vergleicht, vergleicht in Wahrheit statistisches Rauschen mit stabilen Messungen 
        und zieht voreilige Schlüsse über die tatsächliche medizinische Qualität.
        """
    )

    st.markdown("## Die Simulation")
    st.markdown(
        """
        Da wir in der Realität die „wahre“ Qualität eines Krankenhauses nie exakt kennen, beantworten wir 
        die Frage über eine kontrollierte **Datensimulation**: Wir erzeugen ein Gesundheitssystem mit 
        **20 Krankenhäusern**, deren Fallzahlen bewusst extrem variieren:

        - **Sehr kleine Häuser:** 5 bis 20 Patienten pro Jahr (hohes Zufallsrauschen)
        - **Mittlere Häuser:** 30 bis 150 Patienten pro Jahr
        - **Große Unikliniken:** 200 bis 1.000 Patienten pro Jahr (hohe Präzision)

        Für jedes Krankenhaus zieht die Simulation eine **wahre Sterbewahrscheinlichkeiten** ($p_{\\text{true}}$) 
        aus einer Beta-Verteilung mit einem Mittelwert von ca. **26,7 %**. Aus dieser Grundwahrheit werden 
        schließlich die beobachteten Todesfälle simuliert.
        """
    )

    st.markdown(
        """
        **Warum die Simulation unser Goldstandard ist:** Weil wir die Daten selbst generiert haben, kennen 
        wir die echte **Ground Truth** ($p_{\\text{true}}$). In der echten Praxis ist dieser Wert unsichtbar. 
        Hier erlaubt er uns jedoch etwas Entscheidendes: Wir können verschiedene statistische Schätzmethoden 
        (No Pooling vs. Bayesianisches Shrinkage) direkt daran messen, **wie nah sie der tatsächlichen Wahrheit kommen**.
        """
    )

# =================================================================== 2 Daten & EDA
with tab_daten:
    st.markdown("## Explorative Datenanalyse (EDA)")
    st.markdown(
        """
        **Was sehen wir in den Rohdaten?**
        - **`n_patients`**: Fallzahl pro Krankenhaus (5 bis 1000 Patienten)
        - **`deaths`**: Beobachtete Todesfälle
        - **`p_observed`**: Naive Sterberate ($\frac{\text{deaths}}{n}$)
        - **`p_true`**: Echte Sterbewahrscheinlichkeit (Ground Truth)
        """
    )
    
    st.dataframe(df_obs, use_container_width=True)

    # Visueller Vergleich der Stichprobengröße
    st.markdown("### Verteilung der Fallzahlen")
    st.bar_chart(df_obs.set_index("hospital")["n_patients"])


# =================================================================== 3 Naive Analyse
with tab_naiv:
    st.markdown("## Naive Analyse (No Pooling)")
    st.markdown(
        """
        Der einfachste Ansatz berechnet die Sterberate direkt als $\hat{p} = \frac{\text{Todesfälle}}{\text{Patienten}}$.

        **Wo er in die Irre führt:**
        Bei kleinen Häusern ($n=5$) führt ein einzelner Zufallstodesfall sofort zu einer extremen Sterberate (z. B. 20 % oder 40 %). 
        Kleine Häuser belegen dadurch fälschlicherweise immer die Spitzen- und Schlusslichter im Ranking, obwohl ihre *wahre* Qualität unauffällig ist.
        """
    )
    
    # Visueller Vergleich der naiven Beobachtung vs. Wahrheit
    fig_naiv = go.Figure()
    fig_naiv.add_trace(go.Scatter(x=df_obs["hospital"], y=df_truth["p_true"], mode="markers", name="Ground Truth (Wahrheit)", marker=dict(symbol="star", size=11, color="green")))
    fig_naiv.add_trace(go.Scatter(x=df_obs["hospital"], y=df_obs["p_observed"], mode="markers", name="Beobachtet (Naiv)", marker=dict(size=8, color="red")))
    fig_naiv.update_layout(title="Naive Beobachtung vs. Wahrheit (Hohes Rauschen bei kleinen n)", xaxis_title="Krankenhaus (steigende Fallzahl)", yaxis_title="Sterberate")
    st.plotly_chart(fig_naiv, use_container_width=True)


# =================================================================== 4 Identifikation
with tab_ident:
    st.markdown("## Identifikation & Methodik")
    st.markdown(
        r"""
        ### Das Beta-Binomial-Modell
        Um die wahre Sterberate $p_i$ eines Krankenhauses $i$ zu schätzen, verbinden wir die beobachteten Daten mit einem Prior.

        1. **Likelihood (Beobachtung):**  
           Die Todesfälle $y_i$ bei $n_i$ Patienten folgen einer Binomialverteilung:
           $$y_i \sim \text{Binomial}(n_i, p_i)$$

        2. **Prior (Vorwissen):**  
           Wir nehmen an, dass die Raten aller Krankenhäuser aus einer Beta-Verteilung stammen:
           $$p_i \sim \text{Beta}(\alpha, \beta)$$

        3. **Posterior (Aktualisierte Schätzung):**  
           Durch die Konjugiertheit ergibt sich der Posterior-Mittelwert als gewichtetes Mittel:
           $$\hat{p}_i^{\text{Bayes}} = \frac{y_i + \alpha}{n_i + \alpha + \beta} = \frac{n_i}{n_i + N_0} \cdot \bar{p}_{\text{naiv}} + \frac{N_0}{n_i + N_0} \cdot \mu_{\text{prior}}$$

        **Erkenntnis:**  
        Für große Kliniken ($n_i \gg N_0$) dominiert die Datenmenge $\bar{p}_{\text{naiv}}$. Für kleine Kliniken ($n_i \ll N_0$) zieht der Prior $\mu_{\text{prior}}$ das Ergebnis schützend zur Mitte.
        """
    )


# =================================================================== 5 Ergebnis
with tab_ergebnis:
    st.markdown("## Interaktives Bayes-Modell & Kernerkenntnisse")
    st.markdown(
        """
        Verstelle den Regler $N_0$ (Stärke des Priors / Pseudofallzahl). 
        - Bei **$N_0 = 0$** verlässt sich das Modell rein auf die Beobachtung (No Pooling).
        - Je **höher $N_0$**, desto stärker werden kleine Krankenhäuser zum globalen Mittelwert (ca. 26.7 %) gezogen.
        """
    )

    n_0 = st.slider("Prior-Gewichtung ($N_0$)", min_value=0, max_value=100, value=20)

    # Berechnungen durchführen
    df_res, mse_naiv, mse_bayes = analyse.berechne_bayes(df_obs, df_truth, n_0)

    # Diagramm anzeigen
    st.plotly_chart(
        analyse.erstelle_plot(df_res, df_truth, n_0),
        use_container_width=True,
    )

    # Kennzahlen (MSE)
    col1, col2 = st.columns(2)
    col1.metric("MSE Naiv (Rohdaten)", f"{mse_naiv:.5f}")
    col2.metric("MSE Bayes (Shrinkage)", f"{mse_bayes:.5f}", delta=f"{((mse_bayes - mse_naiv) / mse_naiv) * 100:.1f} % Fehler")

    st.markdown(
        """
        ### Fazit
        1. **Kleine Krankenhäuser sind nicht zwingend schlechter:** Extreme Ausschläge nach oben (oder unten) bei kleinen Fallzahlen sind meist reines Zufallsrauschen (Sampling Noise).
        2. **Bayesianisches Shrinkage reduziert den Schätzfehler:** Durch das Einbeziehen des Priors sinkt der Mean Squared Error (MSE) signifikant.
        3. **Faire Qualitätsvergleiche:** Nur durch Regularisierung/Shrinkage lassen sich Fehlanreize in der Gesundheitspolitik (z. B. unberechtigte Klinikschließungen) vermeiden.
        """
    )


# =================================================================== 6 Limitationen
with tab_grenzen:
    st.markdown("## Limitationen des Modells")
    st.markdown(
        """
        * **Kein Risk-Adjustment (Confounding):** Das Modell berücksichtigt nicht, ob Unikliniken schwerere Fälle behandeln als kleine Landkrankenhäuser.
        * **Gefahr des Over-Shrinkage:** Ein kleines Krankenhaus, das *tatsächlich* eine katastrophale Qualität hat, wird durch einen zu starken Prior maskiert.
        * **Wahl des Priors:** Der Parameter $N_0$ muss sensitiv gewählt werden (z. B. über Empirical Bayes oder Cross-Validation).
        """
    )