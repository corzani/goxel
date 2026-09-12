# Rendering GPU in goxel: stato, limiti e proposte

Questo documento descrive il lavoro fatto sul branch `gpu-pathtracer`, cosa
manca e cosa conviene fare dopo.

Obiettivo: avere in goxel su Linux una resa paragonabile a MagicaVoxel, che
non è utilizzabile senza Windows.

## Stato attuale

### Renderer GPU

Un path tracer progressivo che gira sulla GPU e traccia i raggi direttamente
nei voxel, come alternativa al renderer CPU (yocto) nel pannello Render.

| Parte | Come funziona | File |
|---|---|---|
| Dati | I layer visibili vengono uniti in due texture 3D: un atlante dei tile 16³ non vuoti e una tabella con un texel per tile | `src/pathtracer_gpu.c` |
| Raggi | Attraversamento DDA a due livelli, salta i tile vuoti | `data/shaders/pathtracer.glsl` |
| Materiali | Colore base, metallo, rugosità, emissione, opacità, rifrazione | idem |
| Luci | Sole con dimensione regolabile, e voxel emissivi campionati direttamente | idem |
| Ambiente | Tinta unita, cielo procedurale (Day, Sunset, Night, Overcast) o immagine `.hdr`, `.png`, `.jpg` | idem |
| Vetro | Riflessione e rifrazione di Fresnel; i raggi d'ombra attraversano il vetro colorandosi | idem |
| Camera | Prospettica o ortografica, FOV configurabile, profondità di campo a lente sottile | idem |
| Immagine | Accumulo progressivo in un buffer float, bloom, tone mapping ACES, esposizione | idem |

Le sorgenti di luce piccole (voxel emissivi, sole, zone luminose di un HDR)
vengono campionate direttamente e combinate con i rimbalzi tramite MIS:
senza questo le scene notturne sarebbero piene di rumore.

Il renderer riparte da zero quando cambiano scena, camera, materiali o
impostazioni. Richiede GLSL 3.30: viene verificato all'avvio e, dove manca
(macOS, build GLES2, schede vecchie), resta il renderer CPU.

### Import MagicaVoxel (`.vox`)

- **Materiali:** i voxel emissivi, di vetro, di metallo e "media" finiscono in
  layer dedicati, ognuno col suo materiale goxel.
- **Istanze:** un layer per ogni istanza della scena, con la sua
  trasformazione. Prima ogni modello veniva importato una volta sola, quindi
  gli oggetti ripetuti (strade, alberi, bidoni) finivano nel posto sbagliato.
- **Oggetti nascosti:** le istanze in gruppi o layer nascosti diventano layer
  invisibili.
- **Camere:** tutte le camere salvate nel file, con la prima attiva.
- **Illuminazione:** sole, ambiente, esposizione, bloom, pavimento. Se il
  file usa un'immagine HDR e quel file si trova accanto al modello, viene
  caricato.

### Interfaccia

- **Pannello Render:** motore GPU/CPU, Bounces, Exposure, Bloom, Aperture,
  Focus, ambiente (None, Uniform, Sky con i preset, Image con scelta del
  file), Softness del sole.
- **Pannello Cameras:** campo FOV.
- **Pannello Material:** campo Refraction.

### Salvataggio nel `.gox`

- FOV, apertura e fuoco delle camere, e la rifrazione dei materiali, come
  nuove chiavi nei chunk esistenti.
- Le impostazioni di render (motore, campioni, rimbalzi, esposizione, bloom,
  dimensione del sole, ambiente con il percorso dell'immagine, pavimento) in
  un chunk nuovo, `RNDR`.

### Prestazioni misurate

Su Intel Arc integrata (Meteor Lake):

| Scena | Risoluzione | Campioni | Tempo |
|---|---|---|---|
| Scena di prova | 1920×1056 | 512 | circa 8 s |
| Scena di prova con ambiente HDR | 960×540 | 1024 | 6 s |
| `can.vox`, notte, senza vetro | 1920×1056 | 1024 | circa 76 s |
| `can.vox` completo, vetro e HDR | 960×540 | 512 | 58 s |
| Renderer CPU, per confronto | 960×528 | 512 | circa 65 s |

## Compatibilità dei file

I `.gox` scritti da questa versione devono restare apribili dal goxel
originale, che deve ignorare i dati che non conosce. Regole da rispettare:

- **Non alzare mai** la versione del formato (`VERSION`, oggi 2): il lettore
  rifiuta i file con versione superiore alla propria.
- I **chunk sconosciuti** vengono saltati leggendone la lunghezza, quindi un
  chunk nuovo è sempre sicuro.
- Le **chiavi sconosciute** nei dizionari dei chunk esistenti vengono lette e
  ignorate, quindi aggiungerne è sicuro.
- **Attenzione alla lunghezza dei valori:** il lettore copia i valori in un
  buffer senza controllarne la dimensione. Nei chunk esistenti vanno scritti
  solo valori corti (i nostri sono float da 4 byte). I valori lunghi, come il
  percorso di un'immagine, stanno solo in chunk nuovi, che le vecchie
  versioni saltano senza leggerne il contenuto.

Verificato sul campo: un `.gox` scritto da questa versione, con tutti i campi
e il chunk nuovi, è stato aperto dal goxel originale (commit `33be5c8e`)
senza errori, e riletto da questa versione senza perdere nulla.

## Limiti noti

### Renderer

- **Rumore col vetro:** dietro le vetrine resta granuloso. I raggi d'ombra
  attraversano il vetro senza deviare, quindi la luce che passa è
  approssimata.
- **Prestazioni col vetro:** ogni interfaccia costa due attraversamenti in
  più; le scene con molte vetrine sono circa tre volte più lente.
- **Volumetrico:** niente fumo o nebbia. I materiali "media" di MagicaVoxel
  diventano voxel semitrasparenti.
- **Denoiser:** assente, servono molti campioni per un'immagine pulita.
- **Profondità di campo:** solo in proiezione prospettica.
- **Formati immagine:** `.hdr`, `.png` e `.jpg`; niente `.exr`.

### Import `.vox`

- **Coefficienti stimati:** la potenza dell'emissione, l'apertura della lente
  e gli angoli di sole e camera sono tarati a occhio, non su una specifica.
- **Ambiente:** se l'immagine HDR non si trova accanto al modello, viene
  usata la luce uniforme salvata nel file, che di solito è più chiara.
- **Layer nascosti:** vengono importati come layer invisibili, fedeli al
  file. In `can.vox` l'intero strato "Environment" è nascosto, quindi va
  riattivato a mano nel pannello Layers, un layer per volta.
- **Numero di layer:** una scena complessa produce centinaia di layer
  (`can.vox`: 197). goxel non ha gruppi di layer.
- **Export:** salvando in `.vox` i materiali non vengono riscritti.
- **Non importati:** animazioni, nebbia, parametri dell'atmosfera, colore di
  sfondo.

### Altro

- **Renderer CPU:** non è stato riprovato dopo le ultime modifiche. Ignora
  rifrazione, profondità di campo, bloom e ambiente da immagine.
- **Viewport normale:** ignora la rifrazione; l'import cambia l'intensità del
  sole, quindi con le scene notturne il viewport diventa scuro.

## Proposte, in ordine consigliato

1. **Vetro più veloce e meno rumoroso.** Raggruppare i voxel di vetro
   contigui per evitare interfacce interne, e migliorare la luce trasmessa.
2. **Denoiser.** Intel Open Image Denoise (licenza Apache 2.0) sul risultato
   finale: immagini pulite con pochi campioni. Funziona anche su GPU Intel.
3. **Migliorie all'import.** Unire le istanze per layer MagicaVoxel, così
   l'intero "Environment" si accende con un clic; riscrivere i materiali in
   export; verificare la conversione degli angoli della camera.
4. **Sole legato al preset di cielo,** così scegliendo Night la scena diventa
   davvero notturna senza regolare a mano l'intensità.

## Vincoli

Tutto ciò che entra nel progetto deve essere aperto e senza problemi di
copyright. Niente file, immagini o codice presi da MagicaVoxel: la lettura
del formato `.vox` avviene in base alla specifica pubblica, per
interoperabilità. Le risorse di prova vanno prese da fonti libere (per
esempio gli HDRI CC0 di Poly Haven) e restano fuori dal repository.

## Compilazione e prove

```
scons -j$(nproc) mode=release werror=0
```

- `werror=0` serve perché con GCC 16 il codice già esistente
  (`palette.c`, `utils.c`, quickjs) non passa `-Werror`.
- La build `mode=debug` richiede `libasan` e `libubsan`.
- Per provare il renderer: pannello Render, motore GPU, Start.

Le prove automatiche (apertura di un file, click simulati, screenshot,
confronti, salvataggio e rilettura) sono state fatte con un programma che
riusa il ciclo principale di goxel; sta nella cartella temporanea di lavoro e
non nel repository. Per le prove di compatibilità è stata compilata a parte
una copia del goxel originale.
