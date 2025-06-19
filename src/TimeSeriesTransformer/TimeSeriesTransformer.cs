//Copyright Warren Harding 2025.
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.nn.functional;
using System.Collections.Generic;
using System; // Added for IDisposable

namespace TimeSeriesTransformer
{
    /// <summary>
    /// Provides functionality to train and forecast time series data using a Transformer with TorchSharp.
    /// </summary>
    public class TimeSeriesTransformer
    {
        private static torch.Device _getDevice()
        {
            if (torch.cuda.is_available())
            {
                System.Console.WriteLine("Using CUDA GPU for TorchSharp operations.");
                return torch.device(DeviceType.CUDA);
            }
            else
            {
                System.Console.WriteLine("CUDA GPU not available, using CPU for TorchSharp operations.");
                return torch.device(DeviceType.CPU);
            }
        }

        /// <summary>
        /// Trains a Transformer time series model using historical data.
        /// The input and target tensors are expected to be batched sequences.
        /// </summary>
        /// <param name="historicalDataFeatures">The input features tensor for training, shape (batch_size, sequence_length, input_features).</param>
        /// <param name="historicalDataTargets">The target values tensor for training, shape (batch_size, sequence_length, output_features).</param>
        /// <param name="epochs">The number of training epochs.</param>
        /// <param name="learningRate">The learning rate for the optimizer.</param>
        /// <param name="inputFeatures">The number of input features per time step (e.g., 1 for univariate, >1 for multivariate).</param>
        /// <param name="outputFeatures">The number of output features per time step (e.g., 1 for univariate forecast, >1 for multivariate).</param>
        /// <param name="dModel">The dimension of the model's embeddings and hidden states.</param>
        /// <param name="nHead">The number of attention heads.</param>
        /// <param name="numEncoderLayers">The number of sub-encoder-layers in the encoder.</param>
        /// <param name="dimFeedforward">The dimension of the feedforward network.</param>
        /// <param name="dropout">The dropout value.</param>
        /// <param name="maxLength">Maximum sequence length for positional encoding.</param>
        /// <returns>A trained <see cref="TransformerTimeSeriesModel"/> instance, or null if training inputs are invalid.</returns>
        public TransformerTimeSeriesModel? Train(
            Tensor historicalDataFeatures,
            Tensor historicalDataTargets,
            int epochs = 500, // Increased for better training convergence.
            double learningRate = 0.001,
            long inputFeatures = 1,
            long outputFeatures = 1,
            long dModel = 128, // Increased for larger model capacity.
            long nHead = 8,
            long numEncoderLayers = 4, // Increased for deeper model.
            long dimFeedforward = 1024, // Increased for larger feedforward layers.
            double dropout = 0.2,
            long maxLength = 5000)
        {
            if (historicalDataFeatures is null || historicalDataTargets is null)
            {
                System.Console.WriteLine("Input tensors for training cannot be null.");
                return null;
            }

            // Validate tensor shapes: (batch_size, sequence_length, features)
            if (historicalDataFeatures.ndim != 3 || historicalDataFeatures.shape[2] != inputFeatures)
            {
                System.Console.WriteLine($"Input features tensor should have shape (batch_size, sequence_length, {inputFeatures}). Actual dimensions: {historicalDataFeatures.ndim}, shape: ({string.Join(", ", historicalDataFeatures.shape)})");
                return null;
            }
            if (historicalDataTargets.ndim != 3 || historicalDataTargets.shape[2] != outputFeatures)
            {
                System.Console.WriteLine($"Target tensor should have shape (batch_size, sequence_length, {outputFeatures}). Actual dimensions: {historicalDataTargets.ndim}, shape: ({string.Join(", ", historicalDataTargets.shape)})");
                return null;
            }
            if (historicalDataFeatures.shape[0] != historicalDataTargets.shape[0] ||
                historicalDataFeatures.shape[1] != historicalDataTargets.shape[1])
            {
                System.Console.WriteLine("Batch size and sequence length in features and targets must match.");
                return null;
            }

            torch.Device device = _getDevice();

            // Move input data to the determined device.
            using Tensor featuresOnDevice = historicalDataFeatures.to(device);
            using Tensor targetsOnDevice = historicalDataTargets.to(device);

            // Instantiate Transformer model with specified parameters.
            TransformerTimeSeriesModel model = new TransformerTimeSeriesModel(
                inputFeatures: inputFeatures,
                outputFeatures: outputFeatures,
                dModel: dModel,
                nHead: nHead,
                numEncoderLayers: numEncoderLayers,
                dimFeedforward: dimFeedforward,
                dropout: dropout,
                maxLength: maxLength,
                name: "TimeSeriesTransformerModel"
            );
            model.to(device); // Move model to the determined device.

            // Instantiate optimizer (Adam is a good choice for Transformers).
            optim.Optimizer optimizer = optim.Adam(
                model.parameters(),
                lr: learningRate
            );

            TorchSharp.torch.random.manual_seed(42); // For reproducibility

            model.train(); // Set model to training mode

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                optimizer.zero_grad(); // Zero gradients before each backward pass.
                using (Tensor predictions = model.forward(featuresOnDevice)) // Perform forward pass.
                using (Tensor loss = mse_loss(predictions, targetsOnDevice)) // Compute loss.
                {
                    loss.backward(); // Perform backward pass to compute gradients.

                    // Gradient clipping to prevent exploding gradients.
                    // max_norm specifies the maximum norm of the gradients.
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm: 1.0);

                    optimizer.step(); // Update model parameters.
                }
            }

            model.eval(); // Set model to evaluation mode (e.g., disables dropout).
            return model;
        }

        /// <summary>
        /// Forecasts future values using a trained Transformer time series model autoregressively.
        /// The model predicts the next step, which is then appended to the input sequence
        /// for the next prediction, effectively shifting the window forward.
        /// </summary>
        /// <param name="trainedModel">The pre-trained <see cref="TransformerTimeSeriesModel"/> instance.</param>
        /// <param name="initialInputSequence">The tensor representing the last observed sequence to initiate the forecast,
        /// expected shape (1, sequence_length, input_features). The '1' signifies a batch size of one for inference.</param>
        /// <param name="forecastHorizon">The number of future steps to forecast.</param>
        /// <returns>A tensor containing the forecasted values, shape (forecastHorizon, output_features), or null if inputs are invalid.
        /// Each row in the output tensor corresponds to a single forecasted time step.</returns>
        public Tensor? Forecast(
            TransformerTimeSeriesModel trainedModel,
            Tensor initialInputSequence,
            int forecastHorizon = 1)
        {
            if (trainedModel is null)
            {
                System.Console.WriteLine("Trained model cannot be null for forecasting.");
                return null;
            }
            if (initialInputSequence is null)
            {
                System.Console.WriteLine("Initial input sequence tensor for prediction cannot be null.");
                return null;
            }

            // Ensure initialInputSequence has the expected shape: (1, sequence_length, input_features)
            if (initialInputSequence.ndim != 3 || initialInputSequence.shape[0] != 1)
            {
                System.Console.WriteLine("Initial input sequence for prediction should have shape (1, sequence_length, input_features).");
                return null;
            }
            long sequenceLength = initialInputSequence.shape[1]; // Extract the sequence length from the input.

            if (forecastHorizon <= 0)
            {
                System.Console.WriteLine("Forecast horizon must be greater than 0.");
                return null;
            }

            torch.Device device = _getDevice();

            // Ensure the trained model is on the correct device.
            trainedModel.to(device);

            // Clone the initial input sequence and move it to the determined device.
            // This tensor will initiate the autoregressive process.
            Tensor currentInputSequence = initialInputSequence.to(device).clone();

            List<Tensor> predictionsList = new List<Tensor>();

            using (IDisposable noGradScope = torch.no_grad()) // Disable gradient calculation during inference.
            {
                trainedModel.eval(); // Set model to evaluation mode.

                for (int i = 0; i < forecastHorizon; i++)
                {
                    // Create a new DisposeScope for each iteration to manage temporary tensors generated within the loop.
                    using (IDisposable iterationScope = torch.NewDisposeScope())
                    {
                        // Store the current input sequence to dispose it after forming the new one.
                        Tensor oldInputToDispose = currentInputSequence;

                        // Perform a forward pass. nextSequencePrediction is a temporary tensor.
                        using (Tensor nextSequencePrediction = trainedModel.forward(oldInputToDispose))
                        {
                            // Extract the prediction for the very next time step.
                            // This slice and clone needs to persist outside this iterationScope.
                            // Shape will be (1, output_features) since we take the last time step from batch=1.
                            Tensor nextStepPrediction = nextSequencePrediction[.., -1, ..].clone().MoveToOuterDisposeScope();
                            predictionsList.Add(nextStepPrediction);

                            // Prepare the input sequence for the next iteration (autoregressive step).
                            // Slice the old sequence to remove the oldest element and clone to ensure independent memory.
                            using (Tensor slicedSequence = torch.narrow(oldInputToDispose, 1, 1, sequenceLength - 1).clone())
                            // Unsqueeze the new prediction to add a sequence length dimension of 1 for concatenation.
                            // (1, output_features) becomes (1, 1, output_features)
                            using (Tensor unsqueezedPrediction = nextStepPrediction.unsqueeze(1).clone())
                            {
                                // Form the new current input sequence by concatenating the sliced old sequence and the new prediction.
                                // The result is (1, sequenceLength, input_features)
                                currentInputSequence = torch.cat(new Tensor[] {
                                slicedSequence,
                                unsqueezedPrediction
                                }, dim: 1).MoveToOuterDisposeScope();
                            }
                        }
                        
                        // Dispose the old input sequence that was processed in this iteration.
                        oldInputToDispose.Dispose();
                    } // The iterationScope disposes any temporary tensors created within the loop that were not moved out.
                }

                // Concatenate all individual forecasted steps into a single tensor.
                // `predictionsList` contains tensors of shape (1, output_features)
                // `torch.cat(predictionsList, dim: 0)` will result in (forecastHorizon, output_features)
                Tensor finalForecast = torch.cat(predictionsList, dim: 0);

                // Dispose of all individual step predictions collected in the list.
                foreach (Tensor pred in predictionsList)
                {
                    pred.Dispose();
                }

                currentInputSequence.Dispose(); // Dispose the final state of current input sequence.

                // Ensure the final result is returned and not disposed prematurely.
                return finalForecast.MoveToOuterDisposeScope();
            }
        }
    }
}