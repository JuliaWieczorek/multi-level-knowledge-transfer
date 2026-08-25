# Domknięcie wyników przed publikacją - Windows/PowerShell

Ten dokument opisuje wyłącznie szybkie kroki publikacyjne. Nie uruchamia ponownie
300 modeli temporalnych, pięciu modeli source ani augmentacji LLM. Wszystkie
polecenia są CPU-only; karta AMD nie jest potrzebna.

## 1. Co skopiować

Skopiuj cały katalog `multi-level-knowledge-transfer`, włącznie z `data`,
`outputs`, `src`, `tests`, `config.yaml`, `pyproject.toml` i
`analyze_outputs_for_writing.py`. Nie nadpisuj istniejących katalogów
`outputs/report`, `outputs/report_augmented` ani `outputs/strategy_analysis`.
Są to historyczne artefakty, w tym kosztowny bootstrap B=100.

Szczególnej uwagi wymaga użyty wcześniej target-augmented dataset:

```text
data/processed/esconv_checkpoints_augmented.csv
SHA-256: 8bf2eb871050b1f03ae205314745b83f84498c60973c5f2b416478098174dfcd
```

Pliku nie ma w aktualnej kopii projektu, choć kompletne wyniki treningów są
zapisane. Jeśli znajduje się na komputerze, na którym wykonano augmentację,
skopiuj dokładnie ten plik i jego manifest. Nie generuj go ponownie tylko po to,
aby odtworzyć nazwę - inny tekst oznaczałby inny eksperyment.

## 2. Środowisko

Otwórz PowerShell w katalogu projektu i wykonaj:

```powershell
py -3.12 -m venv .venv-publication
Set-ExecutionPolicy -Scope Process Bypass
.\.venv-publication\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[analysis,dev]"
python -m pytest -q
```

Oczekiwany wynik testów w wersji przygotowanej 25 sierpnia 2026 r. to co
najmniej `30 passed`; testy zależne od opcjonalnych bibliotek neural mogą być
pominięte. Przerwij procedurę, jeśli test zmienionej statystyki, CLI albo analizy
strategii nie przechodzi.

## 3. Kontrola kompletności przed obliczeniami

```powershell
$original = Get-ChildItem -LiteralPath outputs\temporal -Recurse -Filter metrics.csv
$augmented = Get-ChildItem -LiteralPath outputs\temporal_augmented -Recurse -Filter metrics.csv
"original metrics files: $($original.Count)"
"augmented metrics files: $($augmented.Count)"
```

Każda wartość powinna wynosić `150`. Następnie sprawdź dataset augmentowany,
jeżeli został odzyskany:

```powershell
$augmentedData = 'data\processed\esconv_checkpoints_augmented.csv'
if (Test-Path -LiteralPath $augmentedData) {
    Get-FileHash -LiteralPath $augmentedData -Algorithm SHA256
} else {
    Write-Warning 'Brak archiwalnego target-augmented CSV; wyniki pozostają analizowalne z outputs.'
}
```

Hash musi być identyczny z wartością podaną w sekcji 1. Brak pliku nie blokuje
raportowania już zapisanych predykcji i metryk, ale musi zostać ujawniony jako
ograniczenie reprodukowalności.

## 4. Szybkie baseline'y

Poniższe polecenia tworzą sześć niezależnych plików metryk. TF-IDF musi używać
`text_seeker`, ponieważ metodologia nie dopuszcza tekstu supportera do gałęzi
tekstowej.

```powershell
$tasks = 'final_intensity','drop_magnitude'
$models = 'naive','tfidf','initial'
foreach ($task in $tasks) {
    foreach ($model in $models) {
        $args = @(
            '-m','mlkt.cli','baseline',
            '--dataset','esconv',
            '--task',$task,
            '--model',$model,
            '--output-dir','outputs\publication_baselines\original'
        )
        if ($model -eq 'tfidf') {
            $args += @('--text-column','text_seeker')
        }
        python @args
        if ($LASTEXITCODE -ne 0) { throw "Baseline failed: $task / $model" }
    }
}

python -m mlkt.cli outcome-baselines `
  --output-dir outputs\publication_baselines\metadata_100
```

Kontrola liczby plików:

```powershell
(Get-ChildItem outputs\publication_baselines\original -Filter '*_metrics.csv').Count
Import-Csv outputs\publication_baselines\metadata_100\metadata_metrics.csv |
    Group-Object split,task,model | Measure-Object
```

Pierwszy wynik powinien wynosić `6`, a tabela metadanych powinna zawierać 16
wierszy (2 splity x 2 cele x 4 specyfikacje). Typowy czas: 1-5 minut.

## 5. Raporty z przedziałami t-Studenta

Nowe katalogi są celowo inne od historycznych, aby niczego nie nadpisywać:

```powershell
python -m mlkt.cli report `
  --experiments outputs\temporal `
  --output-dir outputs\report_tci

python -m mlkt.cli report `
  --experiments outputs\temporal_augmented `
  --output-dir outputs\report_augmented_tci
```

W obu `report_manifest.json` muszą wystąpić: `runs: 150`, pięć seedów,
checkpointy 10/25/50/75/100% oraz opis Student-t CI. Typowy czas: poniżej
2 minut. Te polecenia czytają zapisane metryki i predykcje; nie trenują modeli.

## 6. Analiza strategii: Wald jako główne CI

Nie uruchamiaj bootstrapu B=1000. B=100 zajęło około trzech godzin, więc B=1000
skalowałoby się orientacyjnie do około 30 godzin. Co ważniejsze, liczba replik
nie zmienia oszacowań, analitycznych wartości p ani korekty FDR. W publikacji
używamy analitycznych 95% CI Walda, a zachowane B=100 traktujemy jako analizę
wrażliwości.

```powershell
python -m mlkt.cli analyze-strategies `
  --retrospective-only `
  --bootstrap-samples 0 `
  --output-dir outputs\strategy_analysis_wald
```

Typowy czas na CPU: 5-15 minut. Oczekiwany plik
`outputs/strategy_analysis_wald/retrospective/retrospective_ordinal_associations.csv`
ma 216 wierszy i kolumny `wald_ci_2.5`, `wald_ci_97.5`,
`wald_odds_ratio_2.5`, `wald_odds_ratio_97.5`. Manifest powinien zawierać
`bootstrap_samples: 0` i `primary_interval: analytic_wald_95`.

Nie uruchamiaj ponownie prospective strategy analysis dla kolejnych seedów.
Użyty multinomial logistic regression jest deterministyczny dla tego zbioru;
zmiana `random_state` nie dostarcza niezależnej niepewności. Istniejący wynik
seed 42 należy opisać jako eksploracyjny.

## 7. Audyt końcowy

```powershell
python analyze_outputs_for_writing.py
Get-Content outputs\article_chapter_analysis\publication_integrity_manifest.json
```

Oczekiwane wartości:

```text
original_temporal_runs: 150
augmented_temporal_runs: 150
publication_baseline_metric_files: 6
metadata_baseline_rows: 16
wald_coefficients: 216
```

Archiwum publikacyjne powinno zawierać także:

- `outputs/article_chapter_analysis/REPORT_PL.md`;
- tabele `00`-`28` w tym samym katalogu;
- oba katalogi `report*_tci`;
- `publication_baselines`;
- `strategy_analysis_wald` oraz historyczne `strategy_analysis` z B=100;
- kod, `config.yaml`, `pyproject.toml` i dokładny commit lub kopię stanu roboczego.

## 8. Kiedy można zakończyć

Procedura jest kompletna, jeżeli testy przechodzą, manifest integralności ma
wartości powyżej, oba raporty podają 150 runów, nie nadpisano historycznych
wyników i zapisano hash target-augmented CSV lub jawny komunikat o jego braku.
Nie są wymagane: retrening neural, neural ceiling, nowa augmentacja, B=1000 ani
source ablation bez instance augmentation.

Jeżeli polecenie zostanie przerwane, usuń wyłącznie jego nowy katalog docelowy
(`report_tci`, `report_augmented_tci`, `strategy_analysis_wald` albo odpowiedni
podkatalog baseline) i uruchom ten krok ponownie. Nie usuwaj `outputs/temporal`,
`outputs/temporal_augmented`, `outputs/source_mtl` ani `outputs/strategy_analysis`.

## 9. Dodatkowa analiza trójklasowa `coarse3`

Ta analiza została dodana po domknięciu głównego eksperymentu. Nie zastępuje
macierzy czteroklasowej. Dla `final_intensity` oraz `drop_magnitude` stosuje
mapowanie `1-2 -> 1`, `3 -> 2`, `4-5 -> 3`. W drugim zadaniu klasy oznaczają
odpowiednio mały, średni i duży spadek. Surowe wartości pozostają w kolumnach
`final_target_raw` i `drop_target_raw`.

W odróżnieniu od raportów i bootstrapu ten krok wymaga ponownego treningu
modeli targetowych. Checkpointy source MTL są wykorzystywane ponownie. Kod
wykrywa tylko CUDA; GPU AMD pod Windows nie przyspieszy treningu bez osobnego,
nieobsługiwanego obecnie backendu. Dlatego najpierw uruchom pilot 100%:

```powershell
python -m mlkt.cli run-matrix `
  --label-scheme coarse3 `
  --checkpoints 100 `
  --output-dir outputs\temporal_coarse3

python -m mlkt.cli report `
  --experiments outputs\temporal_coarse3 `
  --output-dir outputs\report_coarse3_tci
```

Pilot ma 30 przebiegów: 6 wariantów x 5 seedów x 1 checkpoint. Sprawdź, czy
manifest raportu podaje `runs: 30`, `label_schemes: ["coarse3"]` i czy każda
średnia nadal ma `n_seeds: 5`. Nie wybieraj wariantu na podstawie testu; decyzję
o kontynuacji oprzyj na z góry ustalonym pełnym porównaniu wszystkich wariantów
i raportuj pilot jako analizę eksploracyjną.

Jeżeli kontynuujemy, to samo polecenie bez `--checkpoints 100` wznowi katalog,
pominie 30 gotowych przebiegów i wykona pozostałe 120:

```powershell
python -m mlkt.cli run-matrix `
  --label-scheme coarse3 `
  --output-dir outputs\temporal_coarse3

python -m mlkt.cli report `
  --experiments outputs\temporal_coarse3 `
  --output-dir outputs\report_coarse3_tci

python -m mlkt.cli compare-label-schemes `
  --original-report outputs\report_tci `
  --coarse-report outputs\report_coarse3_tci `
  --output-dir outputs\label_scheme_comparison
```

Porównanie tworzy `combined_metrics_by_checkpoint.csv`,
`label_scheme_comparison.csv`, `best_models_by_label_scheme.csv` i manifest z
ostrzeżeniem, że delta macro-F1 jest opisowa: schematy rozwiązują różne zadania.
Do baseline'ów dodaj `--label-scheme coarse3` do sześciu poleceń z sekcji 4 i
zapisz je w osobnym katalogu `outputs\publication_baselines\coarse3`.

Nie aktualizuj liczb w artykule ani doktoracie przed powstaniem kompletnego
raportu. Po treningu do tekstu należy dodać jawnie wtórną analizę granulacji
etykiet, rozkłady trzech klas, baseline'y, macro-F1 z przedziałami Studenta,
macierze pomyłek i zastrzeżenie, że wyższej accuracy nie wolno interpretować
jako automatycznej poprawy rozpoznawania klas rzadkich.
