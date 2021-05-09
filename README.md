# KNN

## Autoři

- Gregušová Sabína, Bc.
- Šamánek Jan, Bc.
- Tulušák Adrián, Bc.

## Popis

Školní projekt pro FIT VUT do předmětu KNN s tématem Interaktivní segmentace obrazu.

Projekt vychází z članku [Inside-Outside Guidance](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Interactive_Object_Segmentation_With_Inside-Outside_Guidance_CVPR_2020_paper.pdf)
A snaží se reprodukovat jeho výsledky.

## Technologie

K implementaci byli využity technologie:

- Python 3
- PyTorch
- Numpy

## Spuštění

Projekt obsahuje 2 soubory, které se spouští.

1. [main.py](./src/main.py) je určen k natrénování modelu a uložení jeho verzí. Tento program požaduje několik argumentů.

- --dest - destinace pro ukládání jednotlivých modelů
- --train - cesta ke složce obsahující trénovací dataset
- --test - cesta ke složce obsahující testovací dataset
- --lr - _learning rate_
- --batch-size - _batch-size_

2. [gui.py](./src/gui.py) je program implementující jednoduché grafické rozhraní pro otestování modelu uživatelem na vlastních obrázcích. Tento program požaduje jeden argument

- _model-path_ cesta k modelu
