---
titel: "Retten kleine Krankenhäuser weniger Leben?"
emoji: "🏥"
mitglieder:
  - "Vorname N."
  - "Vorname N."
kurzbeschreibung: "Eine interaktive Untersuchung von Zufallsrauschen und Bayesian Shrinkage bei kleinen Fallzahlen."
---

## Worum geht es?

In dieser Arbeit untersuchen wir, ob kleine Krankenhäuser tatsächlich ein höheres Behandlungsrisiko aufweisen oder ob beobachtete Qualitätsunterschiede lediglich ein Produkt statistischen Zufallsrauschens sind. 

Kleine Kliniken führen oft nur wenige komplexe Eingriffe pro Jahr durch. Wenn bei einer Fallzahl von beispielsweise fünf Patienten rein zufällig zwei Komplikationen auftreten, schnellt die beobachtete Sterberate sofort auf 40 % hoch. Ein naiver Vergleich dieser Raten führt zu schwerwiegenden Fehlschlüssen in der Gesundheitspolitik, da reine Messunsicherheit mit tatsächlicher medizinischer Qualität verwechselt wird. 

Mithilfe einer kontrollierten Datensimulation von 20 Krankenhäusern und bayesianischen Methoden (Bayesian Shrinkage / Hierarchische Modelle) machen wir interaktiv erfahrbar, wie man trotz extremer Fallzahlschwankungen die wahre Behandlungsqualität ($p_{\text{true}}$) verlässlich schätzen kann.

## Daten & Methode

Welche Daten habt ihr benutzt? Welche Methoden (Regression, DAGs, DoubleML,
…)? Codeblöcke gehen so:

```python
import pandas as pd

daten = pd.read_csv("beispiel.csv")
```

## Ergebnisse

Bilder legt ihr einfach mit in euren Projektordner:

<!-- ![Unser wichtigster Plot](ergebnis.png) -->

## Fazit

Was habt ihr gelernt? Was wäre der nächste Schritt?
