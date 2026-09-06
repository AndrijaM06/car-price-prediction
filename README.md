# Car Price Prediction

Regresioni ML projekat koji predviđa cenu polovnog automobila (`priceUSD`) na
osnovu njegovih karakteristika: marke, modela, godine proizvodnje,
kilometraže, tipa goriva, zapremine motora, menjača, pogona, stanja i
segmenta vozila.

## Tok rada

Projekat prati kompletan regresioni ML workflow, korak po korak:

1. **EDA (istraživačka analiza podataka)** - `notebooks/01_eda.ipynb`
2. **Čišćenje podataka** - `notebooks/02_data_cleaning.ipynb` + `src/data_cleaning.py`
3. **Inženjering karakteristika** - `notebooks/03_feature_engineering.ipynb` + `src/feature_engineering.py`
4. **Pretprocesiranje podataka** - `notebooks/04_data_preprocessing.ipynb` + `src/data_preprocessing.py`
5. **Treniranje modela** - `notebooks/05_model_training.ipynb` + `src/model_training.py`
6. **Evaluacija i poređenje modela** - `notebooks/06_model_evaluation.ipynb` + `src/model_evaluation.py`

## Kako pokrenuti projekat

### 1. Instaliraj potrebne biblioteke

```bash
pip install -r requirements.txt
```

### 2. Pokreni skripte redom (iz korenog foldera projekta)

```bash
python -m src.data_cleaning
python -m src.feature_engineering
python -m src.model_training
python -m src.model_evaluation
```

Svaki skript čita rezultat prethodnog koraka i čuva svoj izlaz u `data/`
ili `models/`, tako da ih treba pokretati ovim redosledom.

## Šta je urađeno u svakom koraku

**EDA** - otkriveni su duplikati (87 redova), nedostajuće vrednosti u
`volume(cm3)`, `drive_unit` i `segment`, nerealna kilometraža (do skoro 10
miliona km), i nazivi kolona sa zagradama.

**Čišćenje podataka** - standardizovani nazivi kolona (snake_case),
uklonjeni duplikati, uklonjeni redovi sa nevalidnom cenom, nerealnom
kilometražom (preko 1.000.000 km), nerealnom godinom proizvodnje (van
opsega 1970-2020) i nerealnom zapreminom motora (van opsega 300-8000 cm³).
Rezultat: 55.585 redova.

**Inženjering karakteristika** - dodate karakteristike: `car_age`
(starost automobila), `mileage_per_year` (prosečna kilometraža po godini),
`engine_volume_liters` (zapremina u litrima), `is_newer_car` i
`is_high_mileage` (binarni indikatori).

**Pretprocesiranje** - numeričke kolone su skalirane (`StandardScaler`)
uz imputaciju medijanom; nominalne kategorijske kolone (`make`,
`fuel_type`, `color`, `transmission`, `drive_unit`, `segment`) kodirane su
pomoću `OneHotEncoder`-a; ordinalna kolona `condition` kodirana je pomoću
`OrdinalEncoder`-a uz redosled `for parts < with damage < with mileage`.
Kolona `model` je namerno izostavljena zbog previše (1000+) jedinstvenih
vrednosti.

**Treniranje i evaluacija** - istrenirana su i upoređena četiri
algoritma: Linear Regression, Decision Tree, Random Forest i Support
Vector Machine (SVR), svi nad istim trening/test skupom (80/20 podela,
`random_state=42`).
`***samo Random Forest je preveliki da se sacuva sa commit pa njega ne gleadamo***`

*(Tačne vrednosti se nalaze u `models/model_comparison_results.csv` posle
pokretanja `src/model_evaluation.py`.)*

**Izabrani finalni model: Decision Tree** (`models/car_price_model.joblib`).

## Poznato ograničenje modela

Najveće greške modela javljaju se kod veoma skupih automobila (preko
100.000$) - takvih automobila ima najmanje u skupu podataka, pa model nema
dovoljno primera da nauči obrazac formiranja cene za taj segment. Za
tipične automobile (ispod 30.000$, što čini veliku većinu skupa podataka)
model je znatno precizniji.

SVM (SVR) je dao iznenađujuće loš rezultat sa podrazumevanim parametrima
(`C`, `epsilon`, `kernel`) - verovatno zbog velikog broja one-hot encoded
kolona (136 posle pretprocesiranja) i širokog raspona cena u podacima. Za
bolje rezultate bio bi potreban dodatni tuning parametara, što izlazi iz
okvira ovog zadatka.

Kolona `model` (marka+model specifičnog vozila) je izostavljena iz ulaznih
karakteristika zbog previsoke kardinalnosti (1000+ jedinstvenih
vrednosti). Ovo znači da model ne razlikuje npr. "BMW 3 series" od "BMW X5"
osim kroz ostale karakteristike (zapremina motora, segment, itd.) - buduće
poboljšanje bi moglo biti frequency ili target encoding za ovu kolonu.
