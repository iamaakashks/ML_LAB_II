import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import wittgenstein as lw

# Load dataset
data = load_iris(as_frame=True)
df = data.frame
df['target'] = data.target_names[data.target]  # Replace 0,1,2 with names

# Split data
train, test = train_test_split(df, test_size=0.3, random_state=42)

# RIPPER Rule Learning
ripper = lw.RIPPER()
ripper.fit(train, class_feat='target', pos_class='setosa')

# Print generated rules
print("\n=== RIPPER Rules ===")
print(ripper.ruleset_)

# Evaluate
acc = ripper.score(test, y='target')
print(f"\nAccuracy on test set: {acc:.2f}")

import numpy as np

def foil_gain(p0, n0, p1, n1):
    """Compute FOIL gain for a candidate literal."""
    if p1 == 0: return 0
    return p1 * (np.log2(p1 / (p1 + n1)) - np.log2(p0 / (p0 + n0)))

def foil_algorithm(data, target_col):
    """Simple FOIL-like learner."""
    rules = []
    remaining = data.copy()

    while True:
        rule_conditions = []
        pos = remaining[remaining[target_col] == 1]
        neg = remaining[remaining[target_col] == 0]
        if len(pos) == 0: break

        while len(neg) > 0:
            best_gain = 0
            best_literal = None
            for col in data.columns:
                if col == target_col: continue
                for val in data[col].unique():
                    subset = remaining[remaining[col] == val]
                    p1 = len(subset[subset[target_col] == 1])
                    n1 = len(subset[subset[target_col] == 0])
                    gain = foil_gain(len(pos), len(neg), p1, n1)
                    if gain > best_gain:
                        best_gain = gain
                        best_literal = (col, val)

            if best_literal is None: break
            col, val = best_literal
            rule_conditions.append((col, val))
            remaining = remaining[remaining[col] == val]
            pos = remaining[remaining[target_col] == 1]
            neg = remaining[remaining[target_col] == 0]

        rule = {"IF": rule_conditions, "THEN": 1}
        rules.append(rule)
        # Remove covered positive examples
        covered = np.ones(len(data), dtype=bool)
        for (col, val) in rule_conditions:
            covered &= (data[col] == val)
        data = data[~((data[target_col] == 1) & covered)]

        if len(data[data[target_col] == 1]) == 0:
            break

    return rules

# Example small binary dataset
toy = pd.DataFrame({
    "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast"],
    "Temp": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Mild"],
    "Play": [0, 0, 1, 1, 1, 0, 1]
})

foil_rules = foil_algorithm(toy, "Play")
print("\n=== FOIL Learned Rules ===")
for rule in foil_rules:
    conditions = " AND ".join([f"{c}={v}" for c,v in rule["IF"]])
    print(f"IF {conditions} THEN Play=Yes")

