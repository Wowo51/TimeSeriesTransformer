//Copyright Warren Harding 2025.
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TimeSeriesTransformer
{
    /// <summary>
    /// A Transformer-based model for time series forecasting, utilizing a Transformer Encoder architecture.
    /// This model maps an input sequence of features to an output sequence of features.
    /// The input is expected to be in the shape (batch_size, sequence_length, input_features).
    /// The output will be in the shape (batch_size, sequence_length, output_features).
    /// </summary>
    public class TransformerTimeSeriesModel : nn.Module<Tensor, Tensor>
    {
        private readonly Linear _inputLinear; // Projects input_features to d_model (embedding dimension)
        private readonly PositionalEncoding _positionalEncoder;
        private readonly TransformerEncoder _transformerEncoder;
        private readonly Linear _outputLinear; // Projects d_model to output_features

        /// <summary>
        /// Initializes a new instance of the <see cref="TransformerTimeSeriesModel"/> class.
        /// </summary>
        /// <param name="inputFeatures">The number of features for each time step in the input sequence.</param>
        /// <param name="outputFeatures">The number of features for each time step in the output sequence (forecast).</param>
        /// <param name="dModel">The dimension of the model's embeddings and internal hidden states in the Transformer. All sub-layers in the model produce outputs of dimension dModel.</param>
        /// <param name="nHead">The number of parallel attention heads in the MultiHeadAttention layers. dModel must be divisible by nHead.</param>
        /// <param name="numEncoderLayers">The number of sub-encoder-layers (TransformerEncoderLayer instances) in the encoder.</param>
        /// <param name="dimFeedforward">The dimension of the feedforward network model in the TransformerEncoderLayer.</param>
        /// <param name="dropout">The dropout value for various layers (e.g., positional encoding, attention, feedforward).</param>
        /// <param name="maxLength">Maximum input sequence length the model is designed to handle for positional encoding.</param>
        /// <param name="name">An optional name for the module.</param>
        public TransformerTimeSeriesModel(
            long inputFeatures,
            long outputFeatures,
            long dModel,
            long nHead,
            long numEncoderLayers,
            long dimFeedforward,
            double dropout,
            long maxLength = 5000,
            string name = "")
            : base(name == "" ? "TransformerTimeSeriesModel" : name)
        {
            // Input layer: Projects raw input features into the model's embedding dimension (dModel).
            _inputLinear = Linear(inputFeatures, dModel);

            // Positional Encoder: Adds positional information to the input embeddings.
            _positionalEncoder = new PositionalEncoding(dModel, maxLength, dropout);

            // Transformer Encoder Layer definition: Defines a single layer of the encoder.
            TransformerEncoderLayer encoderLayer = nn.TransformerEncoderLayer(
                d_model: dModel,
                nhead: nHead,
                dim_feedforward: dimFeedforward,
                dropout: dropout
            );
            
            // Transformer Encoder: Stacks multiple encoder layers.
            // TorchSharp's TransformerEncoder expects batch_first=false (sequence_length, batch_size, d_model) by default.
            _transformerEncoder = nn.TransformerEncoder(encoderLayer, numEncoderLayers);

            // Output layer: Projects the output from the Transformer encoder back to the desired output feature dimension.
            _outputLinear = Linear(dModel, outputFeatures);

            // Register all sub-modules for parameter tracking and proper dispose management by TorchSharp.
            RegisterComponents();
        }

        /// <summary>
        /// Performs the forward pass of the Transformer time series model.
        /// </summary>
        /// <param name="src">The input sequence tensor, expected shape (batch_size, sequence_length, input_features).</param>
        /// <returns>The output sequence tensor, shape (batch_size, sequence_length, output_features).</returns>
        public override Tensor forward(Tensor src)
        {
            // 1. Project input features to match the model's embedding dimension.
            Tensor inputProjected = _inputLinear.forward(src); // (batch_size, sequence_length, d_model)

            // 2. Add positional encoding to introduce sequence order information.
            Tensor srcWithPos = _positionalEncoder.forward(inputProjected); // (batch_size, sequence_length, d_model)
            inputProjected.Dispose(); // inputProjected is no longer needed.

            // Transpose to (sequence_length, batch_size, d_model) as TransformerEncoder expects seq_len as first dim.
            Tensor srcTransposed = srcWithPos.transpose(0, 1);
            srcWithPos.Dispose(); // srcWithPos is no longer needed after creating the transposed view.

            // 3. Pass the sequence through the Transformer Encoder stack.
            Tensor encoderOutputTransposed = _transformerEncoder.forward(srcTransposed, null, null); // (sequence_length, batch_size, d_model)
            srcTransposed.Dispose(); // srcTransposed is no longer needed after passing to forward.

            // Transpose back to (batch_size, sequence_length, d_model)
            Tensor encoderOutput = encoderOutputTransposed.transpose(0, 1); // (batch_size, sequence_length, d_model)
            encoderOutputTransposed.Dispose(); // encoderOutputTransposed is no longer needed after creating the final view.

            // 4. Project the encoder's output to the desired output feature dimension.
            Tensor output = _outputLinear.forward(encoderOutput); // (batch_size, sequence_length, output_features)
            encoderOutput.Dispose(); // encoderOutput is no longer needed.

            return output;
        }
    }
}