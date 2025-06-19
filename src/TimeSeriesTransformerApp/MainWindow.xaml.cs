//Copyright Warren Harding 2025.
using Microsoft.Win32;
using System;
using System.Globalization;
using System.IO;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using TimeSeriesTransformer;
using TorchSharp;
using static TorchSharp.torch;

namespace TimeSeriesTransformerApp
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private double[,]? _csvData;
        private CancellationTokenSource? _cts;

        // chart state
        private readonly List<double> _mseValues = new();
        private double _maxMse = 1e-6;          // avoid div-by-zero
        private const double XStep = 4.0;       // px per epoch

        private TransformerTimeSeriesModel? _trainedModel;

        public MainWindow() { InitializeComponent(); }

        #region menu / buttons
        private void OpenCsv_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new Microsoft.Win32.OpenFileDialog
            {
                Filter = "CSV files (*.csv)|*.csv|All files (*.*)|*.*"
            };
            if (dlg.ShowDialog() != true) return;

            try
            {
                var lines = File.ReadAllLines(dlg.FileName);
                var list = new List<double>();
                for (int i = 1; i < lines.Length; i++) // skip header
                {
                    var parts = lines[i].Split(',');
                    bool isNumeric = double.TryParse(parts[1], out double value); // assuming second column is numeric
                    if (isNumeric)
                    {
                        list.Add(value);
                    }
                    else
                    {
                        Log($"Skipping non-numeric line {i + 1}: {lines[i]}");
                    }
                }
                if (list.Count == 0)
                {
                    MessageBox.Show("No numeric data found.", "CSV", MessageBoxButton.OK, MessageBoxImage.Information);
                    return;
                }

                // Convert 1D list to 2D array [numPoints, 1]
                _csvData = new double[list.Count, 1]; // Change _csvData to double[,] in the field declaration as well
                for (int i = 0; i < list.Count; i++)
                {
                    _csvData[i, 0] = list[i];
                }

                CsvFileLabel.Text = System.IO.Path.GetFileName(dlg.FileName);
                Log($"Loaded {_csvData.GetLength(0):n0} rows."); // Use GetLength(0) for rows
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "CSV error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void Exit_Click(object sender, RoutedEventArgs e) => Close();

        private void StartTrain_Click(object? sender, RoutedEventArgs e)
        {
            if (_csvData == null)
            {
                MessageBox.Show("Load a CSV first.");
                return;
            }
            if (_cts != null) return;           // already running

            // quick parsing / validation
            if (!int.TryParse(EpochsBox.Text, out var epochs) ||
                !double.TryParse(LearnRateBox.Text, out var lr) ||
                !int.TryParse(SeqLenBox.Text, out var seqLen) ||
                !int.TryParse(HorizonBox.Text, out var horizon) ||
                !int.TryParse(DModelBox.Text, out var dModel) ||
                !int.TryParse(NHeadBox.Text, out var nHead) ||
                !int.TryParse(LayersBox.Text, out var layers) ||
                !int.TryParse(FFDimBox.Text, out var ffDim) ||
                !double.TryParse(DropoutBox.Text, out var drop))
            {
                MessageBox.Show("Check numeric fields.");
                return;
            }

            // reset chart & log
            _mseValues.Clear();
            _maxMse = 1e-6;
            ErrorLine.Points.Clear();
            CancelBtn.IsEnabled = true;

            _cts = new CancellationTokenSource();
            TrainAsync(epochs, lr, seqLen, horizon, dModel, nHead, layers, ffDim, drop, _cts.Token);
        }

        private void CancelTrain_Click(object? sender, RoutedEventArgs e) => _cts?.Cancel();
        #endregion

        #region training
        private async void TrainAsync(int epochs, double lr, int seqLen, int horizon,
                                      int dModel, int nHead, int layers, int ffDim, double drop,
                                      CancellationToken token)
        {
            try
            {
                Log($"Training started ({epochs} epochs).");

                // --- 1 prepare tensors (using helper from the unit-test project) ------------
                var (norm, mean, sd) = TimeSeriesTransformerTest.TimeSeriesDataGenerator.NormalizeData(_csvData!);
                var (featLst, targLst) = TimeSeriesTransformerTest.TimeSeriesDataGenerator.CreateSequences(norm, seqLen);
                var (trainX, trainY, _, _) = TimeSeriesTransformerTest.TimeSeriesDataGenerator.SplitData(featLst, targLst, 0.8);

                using var features = torch.stack(trainX, dim: 0);
                using var targets = torch.stack(trainY, dim: 0);
                foreach (var t in featLst) t.Dispose();
                foreach (var t in targLst) t.Dispose();

                var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;

                var model = new TransformerTimeSeriesModel(
                    1, 1, dModel, nHead, layers, ffDim, drop, seqLen).to(device);

                var optim = torch.optim.Adam(model.parameters(), lr: lr);
                model.train();

                // --- 2 training loop --------------------------------------------------------
                for (int ep = 1; ep <= epochs; ep++)
                {
                    token.ThrowIfCancellationRequested();

                    optim.zero_grad();
                    using var yHat = model.forward(features.to(device));
                    using var loss = torch.nn.functional.mse_loss(yHat, targets.to(device));
                    loss.backward();
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm: 1.0);
                    optim.step();

                    var mse = loss.ToDouble();
                    AddChartPoint(ep, mse);
                    Log($"Epoch {ep}/{epochs}  MSE {mse:g5}");

                    await Task.Delay(15, token);        // keep UI responsive
                }

                _trainedModel = model.to(torch.CPU);
                var final = _mseValues.Last();
                Log($"Training finished. Final MSE {final:g5}");
            }
            catch (OperationCanceledException)
            {
                Log("Training cancelled.");
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.ToString(), "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally
            {
                CancelBtn.IsEnabled = false;
                _cts?.Dispose(); _cts = null;
            }
        }
        #endregion

        private void SaveModel_Click(object? sender, RoutedEventArgs e)
        {
            if (_trainedModel == null)
            {
                MessageBox.Show("No trained model in memory.", "Save model",
                                MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            var dlg = new SaveFileDialog
            {
                Filter = "PyTorch model (*.pt)|*.pt|All files (*.*)|*.*",
                FileName = "TimeSeriesTransformer.pt"
            };
            if (dlg.ShowDialog() != true) return;

            try
            {
                _trainedModel.save(dlg.FileName);
                Log($"Model saved to “{dlg.FileName}”.");
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Save failed", MessageBoxButton.OK,
                                MessageBoxImage.Error);
            }
        }

        #region chart / log helpers
        private void AddChartPoint(int epoch, double mse)
        {
            Dispatcher.Invoke(() =>
            {
                _mseValues.Add(mse);
                if (mse > _maxMse) _maxMse = mse;

                double h = ChartCanvas.ActualHeight;
                if (h < 10) h = 320;                       // before first resize

                // recompute all points so y-scale follows current max
                ErrorLine.Points.Clear();
                for (int i = 0; i < _mseValues.Count; i++)
                {
                    //double x = i * XStep;
                    double canvasWidth = ChartCanvas.ActualWidth;
                    if (canvasWidth < 10) canvasWidth = 300; // fallback default

                    double xStep = canvasWidth / Math.Max(1, _mseValues.Count - 1);
                    double x = i * xStep;
                    double yNorm = _mseValues[i] / _maxMse;        // 0..1
                    double y = h - yNorm * (h - 4);                // invert so 0 is bottom
                    ErrorLine.Points.Add(new Point(x, y));
                }

                ChartCanvas.Width = Math.Max(ErrorLine.Points.Count * XStep + 4, ChartCanvas.ActualWidth);
            });
        }

        private void Log(string msg) =>
            Dispatcher.Invoke(() =>
            {
                LogBox.AppendText($"[{DateTime.Now:HH:mm:ss}] {msg}\n");
                LogBox.ScrollToEnd();
            });
        #endregion
    }
}