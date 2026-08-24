import sys
from pathlib import Path
import streamlit as st

# Projektordner in den Python-Pfad einfügen
ORDNER = Path(__file__).parent
if str(ORDNER) not in sys.path:
    sys.path.insert(0, str(ORDNER))

# Eigene Logik & Dozenten-Styling importieren
import analyse  # noqa: E402
from utils.theming import FARBEN, merkkasten  # noqa: E402

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
        """)

    st.markdown("## Die Simulation")
    
    st.markdown(
        """
        Da wir in der Realität die „wahre“ Qualität eines Krankenhauses nie exakt kennen, beantworten wir 
        die Frage über eine kontrollierte **Datensimulation**: Wir erzeugen ein Gesundheitssystem mit 
        **20 Krankenhäusern**, deren Fallzahlen bewusst extrem variieren:

        - **Sehr kleine Häuser:** 5 bis 20 Patienten pro Jahr (hohes Zufallsrauschen)
        - **Mittlere Häuser:** 30 bis 150 Patienten pro Jahr
        - **Große Unikliniken:** 200 bis 1.000 Patienten pro Jahr (hohe Präzision)

        Für jedes Krankenhaus zieht die Simulation eine **wahre Sterbewahrscheinlichkeit** ($p_{\\text{true}}$) 
        aus einer Beta-Verteilung mit einem Mittelwert von ca. **26,7 %**. Aus dieser Grundwahrheit werden 
        schließlich die beobachteten Todesfälle simuliert.
        """)

    st.markdown(
        """
        **Warum die Simulation unser Goldstandard ist:** Weil wir die Daten selbst generiert haben, kennen 
        wir die echte **Ground Truth** ($p_{\\text{true}}$). In der echten Praxis ist dieser Wert unsichtbar. 
        Hier erlaubt er uns jedoch etwas Entscheidendes: Wir können verschiedene statistische Schätzmethoden 
        (No Pooling vs. Bayesianisches Shrinkage) direkt daran messen, **wie nah sie der tatsächlichen Wahrheit kommen**.
        """)