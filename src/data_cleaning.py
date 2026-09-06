"""
Skript za čišćenje sirovog skupa podataka o polovnim automobilima.

Tok obrade (isti princip kao u lekciji o čišćenju podataka):

    raw data
      ↓
    cleaning functions
      ↓
    cleaning pipeline
      ↓
    cleaned dataset

Svaki korak čišćenja je izdvojen u posebnu funkciju, a zatim su svi koraci
povezani u jedan pipeline pomoću pandas metode .pipe().
"""

import re

import pandas as pd

RAW_DATA_PATH = "data/cars.csv"
CLEANED_DATA_PATH = "data/cars_cleaned.csv"

# Realne granice koje koristimo za prepoznavanje nevalidnih vrednosti.
MAX_REALISTIC_MILEAGE_KM = 1_000_000
MIN_REALISTIC_YEAR = 1970
MAX_REALISTIC_YEAR = 2020
MIN_REALISTIC_VOLUME_CM3 = 300
MAX_REALISTIC_VOLUME_CM3 = 8000


def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Pretvara nazive kolona u jednostavan i dosledan snake_case oblik."""

    df = df.copy()

    new_columns = []

    for col in df.columns:
        clean_col = col.strip().lower()

        clean_col = clean_col.replace("(", "_")
        clean_col = clean_col.replace(")", "")
        clean_col = clean_col.replace("-", "_")
        clean_col = clean_col.replace("/", "_")

        clean_col = re.sub(r"\s+", "_", clean_col)
        clean_col = re.sub(r"[^a-z0-9_]", "", clean_col)
        clean_col = re.sub(r"_+", "_", clean_col)
        clean_col = clean_col.strip("_")

        new_columns.append(clean_col)

    df.columns = new_columns

    # Poznata mapiranja za ovaj skup podataka:
    # mileage_kilometers -> mileage_km
    # volume_cm3 ostaje isto, ali priceusd -> price_usd radi čitljivosti
    rename_map = {
        "mileage_kilometers": "mileage_km",
        "priceusd": "price_usd",
    }
    df = df.rename(columns=rename_map)

    return df


def _strip_string_values(df: pd.DataFrame) -> pd.DataFrame:
    """Uklanja višak razmaka sa početka i kraja tekstualnih vrednosti."""

    df = df.copy()

    text_columns = df.select_dtypes(include=["object", "string"]).columns

    for col in text_columns:
        df[col] = df[col].astype("string").str.strip()

    return df


def _clean_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizuje kategorijske kolone: mala slova i uklanjanje razmaka."""

    df = df.copy()

    categorical_columns = [
        "make",
        "model",
        "condition",
        "fuel_type",
        "color",
        "transmission",
        "drive_unit",
        "segment",
    ]

    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().str.lower()

    return df


def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Eksplicitno konvertuje kolone za koje očekujemo da budu numeričke.

    Ovo je zaštitni korak: čak i ako su kolone već numeričke, konverzija
    pomoću pd.to_numeric(errors="coerce") osigurava da će skript ostati
    pouzdan ako se u nekoj narednoj verziji fajla pojave nečiste vrednosti.
    """

    df = df.copy()

    numeric_columns = [
        "price_usd",
        "year",
        "mileage_km",
        "volume_cm3",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Uklanja potpuno identične redove (duplikate)."""

    df = df.copy()
    df = df.drop_duplicates()

    return df


def _remove_rows_with_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    """Uklanja redove kod kojih nedostaje ciljna promenljiva price_usd.

    Bez ciljne vrednosti red ne može biti trening primer, jer ne znamo
    stvarnu cenu sa kojom bismo uporedili predikciju modela.
    """

    df = df.copy()
    df = df.dropna(subset=["price_usd"])

    return df


def _remove_invalid_price_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Uklanja redove sa nevalidnom (nula ili negativnom) cenom."""

    df = df.copy()
    df = df[df["price_usd"] > 0]

    return df


def _remove_invalid_mileage_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Uklanja redove sa nerealno velikom kilometražom.

    Tokom EDA smo primetili redove sa kilometražom preko milion kilometara
    (do skoro 10 miliona), što je fizički nemoguće za putnički automobil i
    najverovatnije je posledica greške pri unosu.
    """

    df = df.copy()
    df = df[
        df["mileage_km"].isna()
        | (df["mileage_km"] <= MAX_REALISTIC_MILEAGE_KM)
    ]

    return df


def _remove_invalid_year_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Uklanja redove sa nerealnom godinom proizvodnje.

    Zadržavamo automobile proizvedene između 1970. i 2020. godine. Stariji
    automobili (oldtimeri) su retki i mogu značajno da naruše obrazac koji
    model pokušava da nauči, jer njihova cena zavisi od potpuno drugih
    faktora (kolekcionarska vrednost) nego cena običnog polovnog automobila.
    """

    df = df.copy()
    df = df[
        df["year"].between(MIN_REALISTIC_YEAR, MAX_REALISTIC_YEAR)
    ]

    return df


def _remove_invalid_volume_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Uklanja redove sa nerealnom zapreminom motora.

    Zapremina van opsega 300-8000 cm3 nije tipična za putnički automobil i
    tretiramo je kao grešku u unosu. Nedostajuće vrednosti ne uklanjamo ovde
    - njih ćemo popuniti kasnije u fazi pretprocesiranja.
    """

    df = df.copy()
    df = df[
        df["volume_cm3"].isna()
        | df["volume_cm3"].between(
            MIN_REALISTIC_VOLUME_CM3, MAX_REALISTIC_VOLUME_CM3
        )
    ]

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Kompletan pipeline za čišćenje sirovog skupa podataka."""

    df_clean = (
        df
        .pipe(_standardize_column_names)
        .pipe(_strip_string_values)
        .pipe(_clean_categorical_values)
        .pipe(_convert_numeric_columns)
        .pipe(_remove_duplicate_rows)
        .pipe(_remove_rows_with_missing_target)
        .pipe(_remove_invalid_price_rows)
        .pipe(_remove_invalid_mileage_rows)
        .pipe(_remove_invalid_year_rows)
        .pipe(_remove_invalid_volume_rows)
        .reset_index(drop=True)
    )

    return df_clean


def main() -> None:
    """Load raw data, clean it, and save the cleaned dataset."""

    print("Loading raw dataset...")
    df_raw = pd.read_csv(RAW_DATA_PATH)
    print(f"Raw dataset shape: {df_raw.shape}")

    print("Cleaning dataset...")
    df_cleaned = clean(df_raw)
    print(f"Cleaned dataset shape: {df_cleaned.shape}")

    print("Saving cleaned dataset...")
    df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)

    print(f"Cleaned dataset saved to: {CLEANED_DATA_PATH}")


if __name__ == "__main__":
    main()
