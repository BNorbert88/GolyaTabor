# JAG Gólyatábor 2020: AI bemutató

Az alábbi binder linkkel buildelhető a projekt. 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BNorbert88/GolyaTabor.git/master)

A közös interaktív munka helyszíne a `foglalkozas.ipynb`, mely egy kényelmes felület a bemutatott eszközök
kipróbálására. Igyekeztünk a forráskódot elrejteni a `src` mappába, mivel ennek a foglalkozásnak nem a kódok képezik
a kardinális részét, hanem a szemlélet, amit a munkafüzeten keresztül át tudunk adni.

## Képek, mint táblázatok

Ebben a részben átnézzük, hogy hogyan kell egy képre gondolni, amikor számítógéppel feldolgozzuk. A kép minden
képkockája (pixel) egy számhármas, mely megadja, hogy a pixel mennyire piros, zöld és kék (RGB).
Ezek az úgynevezett színcsatornák. Megmutatjuk, hogyan kell egy ilyen képet kezelni egyszerű listaműveletekkel.

## Kézírás felismerése

Készítünk egy neurális hálót, mely a klasszikus MNIST adatbázis kézzel írott számjegyeit próbálja felismerni.
Interaktív gyakorlatként készítünk rajzolóprogrammal egy "kézzel írott" számot, melyen leteszteljük a betanított hálót.

## ImageNET Challenge


