//Copyright Warren Harding 2025.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;

namespace TimeSeriesTransformerTest
{
    [TestClass]
    public sealed class TestTransformer
    {
        /// <summary>
        /// Tests the TimeSeriesTransformer model's ability to train and forecast
        /// a synthetic complex waveform generated from sine functions (univariate case).
        /// </summary>
        [TestMethod]
        public void UnivariateTimeSeriesForecastTest()
        {
            // Instantiate the TimeSeriesTransformer
            TimeSeriesTransformer.TimeSeriesTransformer transformerTrainer = new TimeSeriesTransformer.TimeSeriesTransformer();

            // Test Parameters for Univariate Series
            int numPoints = 5000;
            double duration = 30.0;
            int sequenceLength = 60;
            int forecastHorizon = 5;
            double trainRatio = 0.8;
            int numSeries = 1; // Univariate
            long inputFeatures = 1;
            long outputFeatures = 1;
            long dModel = 64; // Can be smaller for univariate
            long nHead = 4;
            long numEncoderLayers = 2; // Can be fewer layers
            long dimFeedforward = 256;
            double dropout = 0.1;
            int epochs = 100;
            double learningRate = 0.001;
            double targetMSE = 0.05;

            // 1. Generate Synthetic Univariate Time Series Data (complex waveform)
            double[,] sineParams = new double[,]
            {
                { 1.0, 0.5, 0.0 }, // Amplitude, Frequency, Phase
                { 0.5, 1.2, Math.PI / 2 },
                { 0.3, 2.0, Math.PI / 4 }
            };
            double[] rawDataUni = TimeSeriesDataGenerator.GenerateComplexWaveform(sineParams, duration, numPoints);

            // Convert to a 2D array [numPoints, 1] for consistency with multivariate normalization
            double[,] rawDataMultiEquivalent = new double[numPoints, 1];
            for (int i = 0; i < numPoints; i++)
            {
                rawDataMultiEquivalent[i, 0] = rawDataUni[i];
            }

            // 2. Normalize Data
            (double[,] normalizedDataMulti, double[] means, double[] stdDevs) = TimeSeriesDataGenerator.NormalizeData(rawDataMultiEquivalent);
            double[,] normalizedData = normalizedDataMulti;

            // 3. Create Input-Target Sequences
            (List<Tensor> featuresList, List<Tensor> targetsList) = TimeSeriesDataGenerator.CreateSequences(normalizedData, sequenceLength);

            if (featuresList.Count < (sequenceLength + forecastHorizon + 10))
            {
                foreach (Tensor t in featuresList) t.Dispose();
                foreach (Tensor t in targetsList) t.Dispose();
                Assert.Inconclusive($"Not enough sequences generated for the test parameters. Generated {featuresList.Count} sequences but need at least {sequenceLength + forecastHorizon + 10}. Consider increasing numPoints ({numPoints}) or decreasing sequenceLength/forecastHorizon.");
                return;
            }

            // 4. Split Data into Training and Testing Sets
            (List<Tensor> trainFeaturesList, List<Tensor> trainTargetsList, List<Tensor> testFeaturesList, List<Tensor> testTargetsList) = TimeSeriesDataGenerator.SplitData(featuresList, targetsList, trainRatio);
            Assert.IsNotNull(trainFeaturesList, "Training features list is null.");
            Assert.IsNotNull(trainTargetsList, "Training targets list is null.");
            Assert.IsNotNull(testFeaturesList, "Test features list is null.");
            Assert.IsNotNull(testTargetsList, "Test targets list is null.");
            Assert.IsTrue(trainFeaturesList.Count > 0, "No training data generated. Adjust numPoints or trainRatio.");
            Assert.IsTrue(testFeaturesList.Count > 0, "No test data generated. Adjust numPoints or trainRatio.");

            // 5. Prepare Tensors for Training
            Tensor? trainFeaturesTensor = null;
            Tensor? trainTargetsTensor = null;
            try
            {
                trainFeaturesTensor = torch.stack(trainFeaturesList, dim: 0);
                trainTargetsTensor = torch.stack(trainTargetsList, dim: 0);
            }
            catch (Exception ex)
            {
                Assert.Fail($"Failed to stack training tensors: {ex.Message}");
            }
            finally
            {
                foreach (Tensor t in trainFeaturesList) t.Dispose();
                foreach (Tensor t in trainTargetsList) t.Dispose();
            }

            // 6. Train the Time Series Transformer Model
            TimeSeriesTransformer.TransformerTimeSeriesModel? trainedModel = null;
            try
            {
                trainedModel = transformerTrainer.Train(
                    historicalDataFeatures: trainFeaturesTensor,
                    historicalDataTargets: trainTargetsTensor,
                    epochs: epochs,
                    learningRate: learningRate,
                    inputFeatures: inputFeatures,
                    outputFeatures: outputFeatures,
                    dModel: dModel,
                    nHead: nHead,
                    numEncoderLayers: numEncoderLayers,
                    dimFeedforward: dimFeedforward,
                    dropout: dropout,
                    maxLength: sequenceLength
                );
            }
            finally
            {
                trainFeaturesTensor?.Dispose();
                trainTargetsTensor?.Dispose();
            }

            Assert.IsNotNull(trainedModel, "Model training failed, trainedModel is null.");

            // 7. Perform Forecasting and Evaluate
            Tensor? initialInputSequenceForForecast = null;
            try
            {
                initialInputSequenceForForecast = testFeaturesList[0].unsqueeze(0);
            }
            catch (ArgumentOutOfRangeException)
            {
                Assert.Fail("Test data is empty. Cannot perform forecast. Consider increasing numPoints or adjusting trainRatio.");
                return;
            }

            int firstTestSequenceStartIndexInOriginalData = trainFeaturesList.Count;
            long actualValuesStartIndex = firstTestSequenceStartIndexInOriginalData + sequenceLength;

            if (actualValuesStartIndex + forecastHorizon > normalizedData.GetLength(0))
            {
                initialInputSequenceForForecast?.Dispose();
                trainedModel?.Dispose();
                foreach (Tensor t in featuresList) t.Dispose();
                foreach (Tensor t in targetsList) t.Dispose();
                Assert.Inconclusive($"Not enough actual data points ({normalizedData.GetLength(0)}) to compare forecast against. Need {actualValuesStartIndex + forecastHorizon} points for comparison.");
                return;
            }

            Tensor? forecastedValuesTensor = null;
            try
            {
                forecastedValuesTensor = transformerTrainer.Forecast(trainedModel, initialInputSequenceForForecast, forecastHorizon);
            }
            finally
            {
                initialInputSequenceForForecast?.Dispose();
            }

            Assert.IsNotNull(forecastedValuesTensor, "Forecast failed, forecastedValues is null.");
            Assert.AreEqual(forecastHorizon, forecastedValuesTensor.shape[0], "Forecasted values tensor has incorrect number of steps (rows).");
            Assert.AreEqual(outputFeatures, forecastedValuesTensor.shape[1], "Forecasted values tensor has incorrect number of output features (columns.");

            double[,] actualFutureNormalizedValuesMulti = new double[forecastHorizon, numSeries];
            for (int r = 0; r < forecastHorizon; r++)
            {
                for (int c = 0; c < numSeries; c++)
                {
                    actualFutureNormalizedValuesMulti[r, c] = normalizedData[actualValuesStartIndex + r, c];
                }
            }

            double[,] forecastedNormalizedValuesMulti = new double[forecastHorizon, outputFeatures];
            float[] forecastedFlattened = forecastedValuesTensor.cpu().data<float>().ToArray();
            for (int r = 0; r < forecastHorizon; r++)
            {
                for (int c = 0; c < outputFeatures; c++)
                {
                    forecastedNormalizedValuesMulti[r, c] = forecastedFlattened[r * outputFeatures + c];
                }
            }

            forecastedValuesTensor.Dispose();

            double[,] actualFutureDenormalizedValues = TimeSeriesDataGenerator.DenormalizeData(actualFutureNormalizedValuesMulti, means, stdDevs);
            double[,] forecastedDenormalizedValues = TimeSeriesDataGenerator.DenormalizeData(forecastedNormalizedValuesMulti, means, stdDevs);

            double mse = CalculateMeanSquaredError(actualFutureDenormalizedValues, forecastedDenormalizedValues);
            System.Console.WriteLine($"Univariate Forecast MSE: {mse:F5} (Target: {targetMSE:F5})");
            Assert.IsTrue(mse < targetMSE, $"Univariate Forecast MSE ({mse:F5}) is higher than target ({targetMSE:F5}).");

            foreach (Tensor t in featuresList) t.Dispose();
            foreach (Tensor t in targetsList) t.Dispose();
            trainedModel?.Dispose();
        }

        /// <summary>
        /// Tests the TimeSeriesTransformer model's ability to train and forecast
        /// synthetic complex waveforms generated from sine functions, including multiple
        /// interrelated time series (multivariate case).
        /// </summary>
        [TestMethod]
        public void MultivariateTimeSeriesForecastTest()
        {
            // Instantiate the TimeSeriesTransformer
            TimeSeriesTransformer.TimeSeriesTransformer transformerTrainer = new TimeSeriesTransformer.TimeSeriesTransformer();
            // Test Parameters
            int numPoints = 5000;
            double duration = 30.0;
            int sequenceLength = 60;
            int forecastHorizon = 5;
            double trainRatio = 0.8;
            int numSeries = 3; // Number of interrelated time series to generate
            long inputFeatures = numSeries; // Number of input features (multivariate data)
            long outputFeatures = numSeries; // Number of output features (multivariate forecast)
            long dModel = 128; // Dimension of the model's embeddings
            long nHead = 4; // Number of attention heads
            long numEncoderLayers = 3; // Number of encoder layers
            long dimFeedforward = 256; // Dimension of the feedforward network
            double dropout = 0.1; // Dropout probability
            int epochs = 100; // Number of training epochs
            double learningRate = 0.001; // Learning rate for the optimizer
            double targetMSE = 0.05; // Acceptable Mean Squared Error for the forecast

            // 1. Generate Synthetic Multivariate Time Series Data with interdependencies
            double[] baseFrequencies = new double[]
            {
                0.5,
                0.75,
                1.0
            };
            double[] amplitudes = new double[]
            {
                1.0,
                0.8,
                0.6
            };
            double[] phases = new double[]
            {
                0.0,
                Math.PI / 3,
                Math.PI / 2
            };
            double interdependenceFactor = 0.3; // How much a series influences the next
            double noiseAmplitude = 0.05;
            double[,] rawDataMulti = TimeSeriesDataGenerator.GenerateMultipleInterrelatedSineWaves(numSeries, numPoints, duration, baseFrequencies, amplitudes, phases, interdependenceFactor, noiseAmplitude);

            // 2. Normalize Data (important for neural networks)
            (double[,] normalizedDataMulti, double[] means, double[] stdDevs) = TimeSeriesDataGenerator.NormalizeData(rawDataMulti);
            double[,] normalizedData = normalizedDataMulti; // For consistent naming with existing logic

            // 3. Create Input-Target Sequences
            // Each input sequence predicts the subsequent sequence.
            (List<Tensor> featuresList, List<Tensor> targetsList) = TimeSeriesDataGenerator.CreateSequences(normalizedData, sequenceLength);

            // Ensure sufficient sequences are generated for a meaningful test
            if (featuresList.Count < (sequenceLength + forecastHorizon + 10)) // Arbitrary buffer
            {
                foreach (Tensor t in featuresList)
                    t.Dispose();
                foreach (Tensor t in targetsList)
                    t.Dispose();
                Assert.Inconclusive($"Not enough sequences generated for the test parameters. Generated {featuresList.Count} sequences but need at least {sequenceLength + forecastHorizon + 10}. Consider increasing numPoints ({numPoints}) or decreasing sequenceLength/forecastHorizon.");
                return; // Exit test early
            }

            // 4. Split Data into Training and Testing Sets
            (List<Tensor> trainFeaturesList, List<Tensor> trainTargetsList, List<Tensor> testFeaturesList, List<Tensor> testTargetsList) = TimeSeriesDataGenerator.SplitData(featuresList, targetsList, trainRatio);
            Assert.IsNotNull(trainFeaturesList, "Training features list is null.");
            Assert.IsNotNull(trainTargetsList, "Training targets list is null.");
            Assert.IsNotNull(testFeaturesList, "Test features list is null.");
            Assert.IsNotNull(testTargetsList, "Test targets list is null.");
            Assert.IsTrue(trainFeaturesList.Count > 0, "No training data generated. Adjust numPoints or trainRatio.");
            Assert.IsTrue(testFeaturesList.Count > 0, "No test data generated. Adjust numPoints or trainRatio.");

            // 5. Prepare Tensors for Training (stack lists into single batched Tensors)
            Tensor? trainFeaturesTensor = null;
            Tensor? trainTargetsTensor = null;
            try
            {
                trainFeaturesTensor = torch.stack(trainFeaturesList, dim: 0); // Shape: (batch_size, sequence_length, numSeries)
                trainTargetsTensor = torch.stack(trainTargetsList, dim: 0); // Shape: (batch_size, sequence_length, numSeries)
            }
            catch (Exception ex)
            {
                Assert.Fail($"Failed to stack training tensors: {ex.Message}");
            }
            finally
            {
                // Dispose individual Tensors in the training lists as they are now stacked.
                foreach (Tensor t in trainFeaturesList)
                    t.Dispose();
                foreach (Tensor t in trainTargetsList)
                    t.Dispose();
            }

            // 6. Train the Time Series Transformer Model
            TimeSeriesTransformer.TransformerTimeSeriesModel? trainedModel = null;
            try
            {
                trainedModel = transformerTrainer.Train(historicalDataFeatures: trainFeaturesTensor, historicalDataTargets: trainTargetsTensor, epochs: epochs, learningRate: learningRate, inputFeatures: inputFeatures, outputFeatures: outputFeatures, dModel: dModel, nHead: nHead, numEncoderLayers: numEncoderLayers, dimFeedforward: dimFeedforward, dropout: dropout, maxLength: sequenceLength // Max length for positional encoding should cover max possible sequence length
                );
            }
            finally
            {
                // Dispose training tensors as they are no longer needed after training.
                trainFeaturesTensor?.Dispose();
                trainTargetsTensor?.Dispose();
            }

            Assert.IsNotNull(trainedModel, "Model training failed, trainedModel is null.");

            // 7. Perform Forecasting and Evaluate
            // Select the first sequence from the test set as the initial input for forecasting.
            // This tensor is currently (sequenceLength, numFeatures). Unsqueeze it to (1, sequenceLength, numFeatures) for the Forecast method.
            Tensor? initialInputSequenceForForecast = null;
            try
            {
                initialInputSequenceForForecast = testFeaturesList[0].unsqueeze(0);
            }
            catch (ArgumentOutOfRangeException)
            {
                Assert.Fail("Test data is empty. Cannot perform forecast. Consider increasing numPoints or adjusting trainRatio.");
                return; // Exit test early
            }

            // Determine the starting index in the original 'normalizedData' array for obtaining actual future values.
            // Each sequence `i` in `featuresList` starts at `normalizedData[i]`.
            // The first test sequence starts at the index in 'featuresList' immediately following the training data end.
            int firstTestSequenceStartIndexInOriginalData = trainFeaturesList.Count;
            // The actual values for comparison are in the 'normalizedData' array.
            // The Forecast method uses 'initialInputSequenceForForecast' (which covers `normalizedData[firstTestSequenceStartIndexInOriginalData ... firstTestSequenceStartIndexInOriginalData + sequenceLength - 1]`).
            // It then forecasts `forecastHorizon` steps immediately following this input sequence.
            // So, actual values start from `firstTestSequenceStartIndexInOriginalData + sequenceLength`.
            long actualValuesStartIndex = firstTestSequenceStartIndexInOriginalData + sequenceLength;
            // Verify if there are enough actual points in normalizedData to compare against the forecast horizon.
            if (actualValuesStartIndex + forecastHorizon > normalizedData.GetLength(0))
            {
                // Dispose of Tensors created explicitly within the scope before asserting inconclusive and returning.
                initialInputSequenceForForecast?.Dispose();
                trainedModel?.Dispose();
                // Dispose of all Tensors managed by featuresList/targetsList which were not yet disposed
                foreach (Tensor t in featuresList)
                    t.Dispose();
                foreach (Tensor t in targetsList)
                    t.Dispose();
                Assert.Inconclusive($"Not enough actual data points ({normalizedData.GetLength(0)}) to compare forecast against. Need {actualValuesStartIndex + forecastHorizon} points for comparison.");
                return; // Exit test early
            }

            Tensor? forecastedValuesTensor = null;
            try
            {
                forecastedValuesTensor = transformerTrainer.Forecast(trainedModel, initialInputSequenceForForecast, forecastHorizon);
            }
            finally
            {
                initialInputSequenceForForecast?.Dispose(); // Dispose the input tensor after use
            }

            Assert.IsNotNull(forecastedValuesTensor, "Forecast failed, forecastedValues is null.");
            Assert.AreEqual(forecastHorizon, forecastedValuesTensor.shape[0], "Forecasted values tensor has incorrect number of steps (rows).");
            Assert.AreEqual(outputFeatures, forecastedValuesTensor.shape[1], "Forecasted values tensor has incorrect number of output features (columns.");

            // Extract the actual future normalized values from the original normalized data array.
            double[,] actualFutureNormalizedValuesMulti = new double[forecastHorizon, numSeries];
            for (int r = 0; r < forecastHorizon; r++)
            {
                for (int c = 0; c < numSeries; c++)
                {
                    actualFutureNormalizedValuesMulti[r, c] = normalizedData[actualValuesStartIndex + r, c];
                }
            }

            // Convert forecasted values from Tensor (forecastHorizon, outputFeatures) to a double[,]
            double[,] forecastedNormalizedValuesMulti = new double[forecastHorizon, outputFeatures];
            float[] forecastedFlattened = forecastedValuesTensor.cpu().data<float>().ToArray();
            for (int r = 0; r < forecastHorizon; r++)
            {
                for (int c = 0; c < outputFeatures; c++)
                {
                    forecastedNormalizedValuesMulti[r, c] = forecastedFlattened[r * outputFeatures + c];
                }
            }

            // Dispose forecasted values tensor after converting to array
            forecastedValuesTensor.Dispose();

            // Denormalize both actual and forecasted values back to original scale
            double[,] actualFutureDenormalizedValues = TimeSeriesDataGenerator.DenormalizeData(actualFutureNormalizedValuesMulti, means, stdDevs);
            double[,] forecastedDenormalizedValues = TimeSeriesDataGenerator.DenormalizeData(forecastedNormalizedValuesMulti, means, stdDevs);

            // Calculate Mean Squared Error (MSE) across all features
            double mse = CalculateMeanSquaredError(actualFutureDenormalizedValues, forecastedDenormalizedValues);
            System.Console.WriteLine($"Multivariate Forecast MSE: {mse:F5} (Target: {targetMSE:F5})");
            Assert.IsTrue(mse < targetMSE, $"Multivariate Forecast MSE ({mse:F5}) is higher than target ({targetMSE:F5}).");

            // Final Tensor Disposals:
            foreach (Tensor t in featuresList)
                t.Dispose();
            foreach (Tensor t in targetsList)
                t.Dispose();
            trainedModel?.Dispose();
        }

        /// <summary>
        /// Calculates the Mean Squared Error between two 2D arrays of values.
        /// </summary>
        /// <param name = "actual">The 2D array of actual values.</param>
        /// <param name = "predicted">The 2D array of predicted values.</param>
        /// <returns>The calculated Mean Squared Error, or NaN if arrays have different dimensions.</returns>
        private static double CalculateMeanSquaredError(double[,] actual, double[,] predicted)
        {
            int rows = actual.GetLength(0);
            int cols = actual.GetLength(1);
            if (rows != predicted.GetLength(0) || cols != predicted.GetLength(1))
            {
                System.Console.WriteLine("Error: Actual and predicted arrays must have the same dimensions for MSE calculation.");
                return double.NaN;
            }

            double sumSquaredError = 0.0;
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    sumSquaredError += Math.Pow(actual[r, c] - predicted[r, c], 2);
                }
            }

            return sumSquaredError / (rows * cols);
        }
    }
}