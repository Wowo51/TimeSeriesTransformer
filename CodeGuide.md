# TimeSeriesTransformer – Code Guide

> **Version:** June 19 2025
> **Solution root:** `src/TimeSeriesTransformer.sln`

---

## 1  Introduction

TimeSeriesTransformer is a **.NET 9** solution for training and deploying Transformer‑based models for univariate or multivariate time‑series forecasting.
The codebase is designed to be:

* **Educational** – clear separation between model, training loop, tests, and GUI.
* **Production‑ready** – uses TorchSharp‑CUDA for GPU acceleration, explicit tensor‑lifecycle handling, gradient clipping, and model export.
* **Interactive** – a WPF desktop app lets you load CSV data, tweak hyper‑parameters, visualise training error, and save the trained model.

This guide explains the solution structure, core classes, build instructions, and typical workflows for developers and data‑scientists.

---

## 2  Solution Layout

```
TimeSeriesTransformer.sln
├─ TimeSeriesTransformer/          # Core library (TorchSharp model + trainer)
│  ├─ PositionalEncoding.cs
│  ├─ TransformerTimeSeriesModel.cs
│  ├─ TimeSeriesTransformer.cs     # High‑level Train / Forecast façade
│  └─ TimeSeriesTransformer.csproj
├─ TimeSeriesTransformerTest/      # MSTest project & synthetic‑data helpers
│  ├─ TimeSeriesDataGenerator.cs
│  ├─ TestTransformer.cs
│  ├─ MSTestSettings.cs
│  └─ TimeSeriesTransformerTest.csproj
├─ TimeSeriesTransformerApp/       # WPF training UI
│  ├─ MainWindow.xaml(+.cs)
│  ├─ App.xaml(+.cs)
│  └─ TimeSeriesTransformerApp.csproj
└─ License.txt                     # TranscendAI.tech Non‑Commercial license
```

---

## 3  Core Library (`TimeSeriesTransformer`)

### 3.1  Key Responsibilities

* Implement an **encoder‑only Transformer** for sequence‑to‑sequence regression.
* Provide **training** (`Train`) and **autoregressive forecasting** (`Forecast`) helpers.
* Manage **device selection** (CUDA → CPU fallback) and safe tensor disposal via `DisposeScope`.

### 3.2  Important Classes

| Class                                                  | Purpose                                                                                                                                     |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **`PositionalEncoding`**                               | Pre‑computes sinusoidal encodings and adds them to embeddings. Registered as a *buffer* so the tensor moves with the model between CPU/GPU. |
| **`TransformerTimeSeriesModel`**                       | Wraps:                                                                                                                                      |
| `Linear→PositionalEncoding→TransformerEncoder→Linear`. |                                                                                                                                             |
| Input/Output shapes: **(batch, seqLen, features)**.    |                                                                                                                                             |
| **`TimeSeriesTransformer`**                            | High‑level API. Handles:                                                                                                                    |

1. Validation & device transfer.
2. Training loop with Adam, MSE loss, gradient‑clipping, reproducible seed.
3. Autoregressive forecasting by sliding a window and re‑feeding predictions. |

#### 3.3  Memory & Device Management

TorchSharp does not have Python‑like automatic tensor GC.
The code therefore:

* Wraps temporary operations in `using var scope = torch.NewDisposeScope()`.
* Calls `tensor.Dispose()` where needed.
* Uses `.MoveToOuterDisposeScope()` to keep final results alive.

This pattern prevents CUDA OOMs and leaks when looping through many epochs.

---

## 4  Test Suite (`TimeSeriesTransformerTest`)

### 4.1  Synthetic Data Utilities

`TimeSeriesDataGenerator` creates:

* **Complex waveforms** (sum of sine waves).
* **Inter‑related multivariate series** with controllable noise & coupling.
* Helpers for Z‑score **normalisation / de‑normalisation** and sequence windowing.

### 4.2  Unit Tests (`TestTransformer.cs`)

Two MSTest methods verify end‑to‑end accuracy:

1. **UnivariateTimeSeriesForecastTest**
   Trains on a single complex waveform and asserts forecast *MSE < 0.05*.
2. **MultivariateTimeSeriesForecastTest**
   Generates 3 correlated series and checks multivariate forecast error.

Run them with:

```bash
cd src/TimeSeriesTransformerTest
dotnet test -c Release
```

---

## 5  Desktop App (`TimeSeriesTransformerApp`)

A minimal WPF front‑end for non‑coders.

### 5.1  Workflow

1. **File → Open CSV…**
   *CSV must have a numeric column in position 1; header row is skipped.*
2. **Set Training Parameters** in the left panel (epochs, model size, etc.).
3. **Start Training**
   *Live MSE* is plotted on the right; training can be cancelled.
4. **File → Save Model…** exports a `.pt` file usable by Torch/TorchSharp.

### 5.2  Chart Scaling

The controller recomputes Y‑range each update so the curve always fits. X‑spacing is dynamic based on canvas width.

---

## 6  Building & Running

### 6.1  Prerequisites

* **Visual Studio 2022 17.14+** or `dotnet 9 SDK` preview.
* **CUDA 11.8+** & NVIDIA driver if using `TorchSharp-cuda-windows`.

### 6.2  Restore & Build

```bash
git clone <repo_url>
cd TimeSeriesTransformer/src
dotnet restore
dotnet build -c Release
```

### 6.3  Run Tests

```bash
dotnet test TimeSeriesTransformerTest -c Release
```

### 6.4  Launch WPF App

```bash
dotnet run --project TimeSeriesTransformerApp -c Release
```

---

## 7  Using the Library in Your Code

```csharp
using TorchSharp;
using TimeSeriesTransformer;

// 1.  Prepare Float32 tensors of shape (batch, seqLen, features)
Tensor features = /* load or generate */;
Tensor targets  = /* same shape */;

var trainer = new TimeSeriesTransformer.TimeSeriesTransformer();
var model = trainer.Train(features, targets,
                          epochs: 200, learningRate: 1e-3,
                          inputFeatures: features.shape[2],
                          outputFeatures: targets.shape[2]);

// 2.  Forecast 10 future steps
Tensor lastWindow = /* (1, seqLen, features) */;
Tensor forecast   = trainer.Forecast(model, lastWindow, forecastHorizon:10);

// 3.  Persist model
model.save("MyModel.pt");
```

---

## 8  Extending & Customising

* **Different sequence lengths** – `maxLength` in `PositionalEncoding` & model constructor must cover the longest sequence you will pass.
* **Loss functions** – swap `mse_loss` for MAE, Huber, etc.
* **Decoder‑style forecasting** – add a TransformerDecoder if you need teacher‑forcing or multi‑step outputs at once.
* **Python interoperability** – the saved `.pt` file can be loaded in PyTorch:

  ```python
  import torch
  model = torch.load('TimeSeriesTransformer.pt')
  model.eval()
  ```

---

## 9  Performance Tips

* **GPU** – Training 5 000 sequences × 100 epochs fits easily in 8 GB VRAM with default dimensions.
* **Batch size** – Increase by stacking more sequences before calling `Train`. Ensure memory allows.
* **Dispose** – If writing custom loops, mimic the `DisposeScope` pattern.

---

## 10  Troubleshooting

| Symptom                            | Likely Cause                                            | Fix                                                               |
| ---------------------------------- | ------------------------------------------------------- | ----------------------------------------------------------------- |
| `torch.Device` has no `cuda`/`cpu` | TorchSharp namespace clash                              | Ensure `using static TorchSharp.torch` and **TorchSharp 0.105+**. |
| `CUDA out of memory`               | Batch/seqLen too large                                  | Reduce dimensions **or** use CPU.                                 |
| `Forecast shape mismatch`          | Initial input sequence shape != `(1, seqLen, features)` | `unsqueeze(0)` the sample window.                                 |
| Blank chart in WPF                 | Canvas width/height < 10 at first measure               | Resize window or start training to trigger redraw.                |

---

## 11  License

The project ships under the **TranscendAI.tech Non‑Commercial License v1.0** (see `License.txt`).
Contact TranscendAI.tech for commercial terms.

---

## 12  Further Reading

* Vaswani et al. "Attention Is All You Need" (2017) – original Transformer paper.
* TorchSharp documentation [https://github.com/dotnet/TorchSharp](https://github.com/dotnet/TorchSharp).
* Microsoft MSTest [https://learn.microsoft.com/dotnet/core/testing/](https://learn.microsoft.com/dotnet/core/testing/).

---

</br>
Copyright [TranscendAI.tech](https://TranscendAI.tech) 2025.<br>
Authored by Warren Harding. AI assisted.</br>
