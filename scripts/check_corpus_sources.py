import pandas as pd
from pathlib import Path

main_path = Path("data/processed/06_nmt_filtered/train.csv")
xl_path = Path("data/processed/06_nmt_filtered_xl/train.csv")

print("=== MAIN (corpus original 3204 pares) ===")
if main_path.exists():
    df = pd.read_csv(main_path, encoding="utf-8-sig")
    print("total rows:", len(df))
    print(df["origin_source"].value_counts())
else:
    print("no existe")

print()
print("=== XL (corpus expandido) ===")
df = pd.read_csv(xl_path, encoding="utf-8-sig")
print("total rows:", len(df))
print(df["origin_source"].value_counts())
