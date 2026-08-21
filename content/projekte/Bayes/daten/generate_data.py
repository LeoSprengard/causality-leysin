import numpy as np
import pandas as pd
from pathlib import Path

# Reproduzierbarkeit
rng = np.random.default_rng(42)

# ------------------------------------------------------------
# 1. Fallzahlen der 20 Krankenhäuser
#    Bewusst stark unterschiedlich: von sehr kleinen bis
#    zu großen Krankenhäusern.
# ------------------------------------------------------------
n_patients = np.array([
    5, 8, 12, 15, 20,
    30, 40, 50, 70, 90,
    120, 150, 200, 250, 300,
    400, 500, 600, 800, 1000
])

hospital = [f"H{i:02d}" for i in range(1, 21)]

# ------------------------------------------------------------
# 2. Wahre Sterbewahrscheinlichkeiten
#
#    Die Krankenhäuser unterscheiden sich tatsächlich.
#    Beta(8, 22) hat einen Mittelwert von ca. 26.7 %.
#    Diese Werte kennen wir nur, weil wir die Daten simulieren.
# ------------------------------------------------------------
p_true = rng.beta(8, 22, size=20)

# ------------------------------------------------------------
# 3. Todesfälle simulieren
# ------------------------------------------------------------
deaths = rng.binomial(n_patients, p_true)

# ------------------------------------------------------------
# 4. Beobachtete Sterberate berechnen
# ------------------------------------------------------------
p_observed = deaths / n_patients

# ------------------------------------------------------------
# 5. Beobachtbare Daten speichern
#    Diese Datei würde später die Grundlage für unsere
#    statistische Analyse bilden.
# ------------------------------------------------------------
observed = pd.DataFrame({
    "hospital": hospital,
    "n_patients": n_patients,
    "deaths": deaths,
    "p_observed": p_observed
})

# ------------------------------------------------------------
# 6. Wahrheit separat speichern
#    Diese Datei verwenden wir später nur, um zu überprüfen,
#    wie gut unsere Modelle die tatsächlichen Werte schätzen.
# ------------------------------------------------------------
truth = pd.DataFrame({
    "hospital": hospital,
    "p_true": p_true
})

output_dir = Path(__file__).resolve().parent / "data"
output_dir.mkdir(exist_ok=True)

observed_path = output_dir / "hospitals.csv"
truth_path = output_dir / "simulation_truth.csv"

observed.to_csv(observed_path, index=False)
truth.to_csv(truth_path, index=False)

print("Daten erfolgreich generiert.")
print(f"Beobachtbare Daten: {observed_path}")
print(f"Wahre Werte:        {truth_path}")
print()
print(observed.to_string(index=False))
