//Copyright Warren Harding 2025.
using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;

namespace TimeSeriesTransformerTest
{
    /// <summary>
    /// Provides utility methods for generating and processing time series data for testing purposes.
    /// </summary>
    public static class TimeSeriesDataGenerator
    {
        /// <summary>
        /// Generates a single sine wave.
        /// </summary>
        /// <param name="amplitude">The amplitude of the sine wave.</param>
        /// <param name="frequency">The frequency of the sine wave (cycles per unit duration).</param>
        /// <param name="phase">The phase offset of the sine wave in radians.</param>
        /// <param name="duration">The total duration of the time series.</param>
        /// <param name="numPoints">The number of data points to generate over the duration.</param>
        /// <returns>A double array representing the generated sine wave.</returns>
        public static double[] GenerateSineWave(double amplitude, double frequency, double phase, double duration, int numPoints)
        {
            double[] data = new double[numPoints];
            double timeStep = duration / numPoints;
            for (int i = 0; i < numPoints; i++)
            {
                data[i] = amplitude * Math.Sin(2 * Math.PI * frequency * (i * timeStep) + phase);
            }
            return data;
        }

        /// <summary>
        /// Generates a complex waveform by summing multiple sine waves.
        /// </summary>
        /// <param name="sineParams">A 2D array where each row contains {amplitude, frequency, phase} for a sine wave.</param>
        /// <param name="duration">The total duration of the time series.</param>
        /// <param name="numPoints">The number of data points to generate over the duration.</param>
        /// <returns>A double array representing the generated complex waveform.</returns>
        public static double[] GenerateComplexWaveform(double[,] sineParams, double duration, int numPoints)
        {
            double[] complexWave = new double[numPoints];
            for (int i = 0; i < sineParams.GetLength(0); i++)
            {
                double amplitude = sineParams[i, 0];
                double frequency = sineParams[i, 1];
                double phase = sineParams[i, 2];
                double[] sineWave = GenerateSineWave(amplitude, frequency, phase, duration, numPoints);
                for (int j = 0; j < numPoints; j++)
                {
                    complexWave[j] += sineWave[j];
                }
            }
            return complexWave;
        }

        /// <summary>
        /// Generates multiple interrelated time series using sine functions and propagating influence.
        /// </summary>
        /// <param name="numSeries">The number of time series to generate.</param>
        /// <param name="numPoints">The number of data points for each series.</param>
        /// <param name="duration">The total duration of the time series.</param>
        /// <param name="baseFrequencies">Array of base frequencies for each series' primary sine component.</param>
        /// <param name="amplitudes">Array of amplitudes for each series' primary sine component.</param>
        /// <param name="phases">Array of phase offsets for each series' primary sine component.</param>
        /// <param name="interdependenceFactor">Factor by which previous series influence the current one.</param>
        /// <param name="noiseAmplitude">Amplitude of random noise added to each series.</param>
        /// <param name="seed">Optional seed for the random number generator for reproducibility.</param>
        /// <returns>A 2D double array where rows are time points and columns are individual time series [numPoints, numSeries].</returns>
        public static double[,] GenerateMultipleInterrelatedSineWaves(
            int numSeries, int numPoints, double duration,
            double[] baseFrequencies, double[] amplitudes, double[] phases,
            double interdependenceFactor, double noiseAmplitude, int? seed = null)
        {
            if (numSeries <= 0 || numPoints <= 0 || duration <= 0)
            {
                throw new ArgumentOutOfRangeException("numSeries, numPoints, and duration must be positive values.");
            }
            if (baseFrequencies.Length < numSeries || amplitudes.Length < numSeries || phases.Length < numSeries)
            {
                throw new ArgumentException("Lengths of baseFrequencies, amplitudes, and phases arrays must be at least numSeries.");
            }

            double[,] data = new double[numPoints, numSeries];
            double timeStep = duration / numPoints;
            Random random = seed.HasValue ? new Random(seed.Value) : new Random();

            for (int s = 0; s < numSeries; s++)
            {
                for (int i = 0; i < numPoints; i++)
                {
                    double currentSineValue = amplitudes[s] * Math.Sin(2 * Math.PI * baseFrequencies[s] * (i * timeStep) + phases[s]);
                    
                    double interdependenceValue = 0.0;
                    if (s > 0)
                    {
                        // Add influence from the previous series
                        interdependenceValue = interdependenceFactor * data[i, s - 1];
                    }

                    double noise = (random.NextDouble() * 2 - 1) * noiseAmplitude; // Random value between -noiseAmplitude and +noiseAmplitude
                    data[i, s] = currentSineValue + interdependenceValue + noise;
                }
            }
            return data;
        }

        /// <summary>
        /// Creates input-target sequences from a 2D time series array (multiple series), suitable for sequence-to-sequence models.
        /// Each input sequence `[x_t, ..., x_{t+L-1}]` is mapped to a target sequence `[x_{t+1}, ..., x_{t+L}]`.
        /// </summary>
        /// <param name="data">The 2D double array representing the time series data [time_points, num_features].</param>
        /// <param name="sequenceLength">The desired length of each input and target sequence.</param>
        /// <returns>A tuple containing lists of TorchSharp Tensors for features (inputs) and targets.</returns>
        public static (List<Tensor> Features, List<Tensor> Targets) CreateSequences(double[,] data, int sequenceLength)
        {
            List<Tensor> features = new List<Tensor>();
            List<Tensor> targets = new List<Tensor>();

            int numPoints = data.GetLength(0);
            int numFeatures = data.GetLength(1);

            // The loop iterates until there are enough points remaining to form both an input and a target sequence.
            // For input data[i .. i+sequenceLength-1], we need target data[i+1 .. i+sequenceLength].
            // This means the last target element data[i+sequenceLength] must exist within 'data'.
            // So, i + sequenceLength must be less than data.GetLength(0).
            for (int i = 0; i < numPoints - sequenceLength; i++)
            {
                double[,] inputChunk = new double[sequenceLength, numFeatures];
                double[,] targetChunk = new double[sequenceLength, numFeatures];

                for (int row = 0; row < sequenceLength; row++)
                {
                    for (int col = 0; col < numFeatures; col++)
                    {
                        inputChunk[row, col] = data[i + row, col];
                        targetChunk[row, col] = data[i + 1 + row, col];
                    }
                }

                // Convert to Tensors. This creates (sequenceLength, numFeatures) tensors.
                Tensor inputTensor = torch.tensor(inputChunk, dtype: ScalarType.Float32);
                Tensor targetTensor = torch.tensor(targetChunk, dtype: ScalarType.Float32);

                features.Add(inputTensor);
                targets.Add(targetTensor);
            }
            return (features, targets);
        }

        /// <summary>
        /// Splits lists of features and targets into training and testing sets based on a specified ratio.
        /// </summary>
        /// <param name="features">The list of feature Tensors.</param>
        /// <param name="targets">The list of target Tensors.</param>
        /// <param name="trainRatio">The ratio of data to be used for training (e.g., 0.8 for 80% training, 20% testing).</param>
        /// <returns>A tuple containing separate lists for training features, training targets, test features, and test targets.</returns>
        public static (List<Tensor> TrainFeatures, List<Tensor> TrainTargets, List<Tensor> TestFeatures, List<Tensor> TestTargets)
            SplitData(List<Tensor> features, List<Tensor> targets, double trainRatio)
        {
            int totalSamples = features.Count;
            int trainSize = (int)(totalSamples * trainRatio);

            List<Tensor> trainFeatures = features.GetRange(0, trainSize);
            List<Tensor> trainTargets = targets.GetRange(0, trainSize);

            List<Tensor> testFeatures = features.GetRange(trainSize, totalSamples - trainSize);
            List<Tensor> testTargets = targets.GetRange(trainSize, totalSamples - trainSize);

            return (trainFeatures, trainTargets, testFeatures, testTargets);
        }

        /// <summary>
        /// Normalizes a 2D double array (multiple time series) using Z-score normalization (mean 0, standard deviation 1) for each feature.
        /// </summary>
        /// <param name="data">The input 2D double array to normalize [time_points, num_features].</param>
        /// <returns>A tuple containing the normalized data, and arrays of means and standard deviations for each feature.</returns>
        public static (double[,] normalizedData, double[] means, double[] stdDevs) NormalizeData(double[,] data)
        {
            int numPoints = data.GetLength(0);
            int numFeatures = data.GetLength(1);

            if (numPoints == 0 || numFeatures == 0)
            {
                return (new double[0, 0], new double[0], new double[0]);
            }

            double[,] normalizedData = new double[numPoints, numFeatures];
            double[] means = new double[numFeatures];
            double[] stdDevs = new double[numFeatures];

            for (int col = 0; col < numFeatures; col++)
            {
                double sum = 0.0;
                for (int row = 0; row < numPoints; row++)
                {
                    sum += data[row, col];
                }
                double mean = sum / numPoints;
                means[col] = mean;

                double sumSqDiff = 0.0;
                for (int row = 0; row < numPoints; row++)
                {
                    sumSqDiff += Math.Pow(data[row, col] - mean, 2);
                }
                double stdDev = Math.Sqrt(sumSqDiff / numPoints);
                stdDevs[col] = stdDev;

                if (stdDev == 0) // Avoid division by zero if all values are the same
                {
                    for (int row = 0; row < numPoints; row++)
                    {
                        normalizedData[row, col] = 0.0;
                    }
                }
                else
                {
                    for (int row = 0; row < numPoints; row++)
                    {
                        normalizedData[row, col] = (data[row, col] - mean) / stdDev;
                    }
                }
            }
            return (normalizedData, means, stdDevs);
        }

        /// <summary>
        /// Denormalizes a 2D double array back to its original scale using the provided means and standard deviations for each feature.
        /// </summary>
        /// <param name="normalizedData">The normalized 2D double array [time_points, num_features].</param>
        /// <param name="means">The array of original means used for normalization, one for each feature.</param>
        /// <param name="stdDevs">The array of original standard deviations used for normalization, one for each feature.</param>
        /// <returns>A 2D double array representing the denormalized data.</returns>
        public static double[,] DenormalizeData(double[,] normalizedData, double[] means, double[] stdDevs)
        {
            int numPoints = normalizedData.GetLength(0);
            int numFeatures = normalizedData.GetLength(1);

            if (numPoints == 0 || numFeatures == 0)
            {
                return new double[0, 0];
            }
            if (means.Length != numFeatures || stdDevs.Length != numFeatures)
            {
                throw new ArgumentException("Lengths of means and stdDevs arrays must match the number of features in normalizedData.");
            }

            double[,] denormalizedData = new double[numPoints, numFeatures];
            for (int col = 0; col < numFeatures; col++)
            {
                for (int row = 0; row < numPoints; row++)
                {
                    denormalizedData[row, col] = normalizedData[row, col] * stdDevs[col] + means[col];
                }
            }
            return denormalizedData;
        }
    }
}