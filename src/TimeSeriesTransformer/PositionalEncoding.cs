//Copyright Warren Harding 2025.
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn.functional;

namespace TimeSeriesTransformer
{
    /// <summary>
    /// Implements positional encoding as described in the "Attention Is All You Need" paper.
    /// This module adds positional information to input embeddings, which is crucial for Transformers
    /// as they do not inherently process sequence order.
    /// </summary>
    public class PositionalEncoding : nn.Module<Tensor, Tensor>
    {
        private readonly float _dropout;
        private readonly Tensor _pe; // This field will hold the registered positional encoding tensor.

        /// <summary>
        /// Initializes a new instance of the <see cref="PositionalEncoding"/> class.
        /// </summary>
        /// <param name="dModel">The dimension of the model (embedding size).</param>
        /// <param name="maxLength">The maximum sequence length for which positional encodings will be pre-computed.</param>
        /// <param name="dropout">The dropout probability applied to the output of the positional encoding.</param>
        /// <param name="name">An optional name for the module.</param>
        public PositionalEncoding(long dModel, long maxLength = 5000, double dropout = 0.1, string name = "")
            : base(name == "" ? "PositionalEncoding" : name)
        {
            this._dropout = (float)dropout;

            Tensor tempPe; // Declare a temporary tensor to build the positional encoding.
            using (var scope = NewDisposeScope())
            {
                // Create a new tensor directly within the scope, initially filled with zeros.
                // This tensor will be moved out of the scope before it's registered.
                tempPe = torch.zeros(1, maxLength, dModel, requires_grad: false);

                using Tensor position = torch.arange(0, maxLength, dtype: ScalarType.Float32).unsqueeze(1);
                using Tensor div_term = torch.exp(torch.arange(0, dModel, 2, dtype: ScalarType.Float32) * (-System.Math.Log(10000.0) / dModel));
                
                // Create a temporary view of `tempPe` without the batch dimension for population.
                // This view (`peBaseView`) will be implicitly disposed by `scope`,
                // but crucially, because `tempPe` is explicitly moved out, its underlying data remains valid.
                Tensor peBaseView = tempPe.squeeze(0);

                using Tensor sinValueTensor = torch.sin(position * div_term);
                using Tensor cosValueTensor = torch.cos(position * div_term);

                using Tensor evenColumnIndices = torch.arange(0, dModel, 2, dtype: ScalarType.Int64);
                using Tensor oddColumnIndices = torch.arange(1, dModel, 2, dtype: ScalarType.Int64);

                // Populate `peBaseView`, which directly modifies the underlying data of `tempPe`.
                peBaseView.index_copy_(1, evenColumnIndices, sinValueTensor);
                peBaseView.index_copy_(1, oddColumnIndices, cosValueTensor);
            
                // Move `tempPe` out of the current DisposeScope.
                // This ensures `tempPe` (and its underlying data) survives the scope's exit.
                tempPe = tempPe.MoveToOuterDisposeScope();
            }
            
            // Assign the now persistent `tempPe` to the class field.
            this._pe = tempPe;

            // Register the completely populated `_pe` tensor as a buffer.
            // This ensures TorchSharp's module system manages its lifecycle,
            // preventing premature disposal during subsequent forward passes.
            this.register_buffer("pe", _pe);
        }

        /// <summary>
        /// Performs the forward pass of the positional encoding.
        /// Adds the pre-computed positional encodings to the input tensor.
        /// </summary>
        /// <param name="x">The input tensor, expected shape (batch_size, sequence_length, d_model).</param>
        /// <returns>The tensor with positional encoding added, after applying dropout.</returns>
        public override Tensor forward(Tensor x)
        {
            // Positional encoding tensor `_pe` is a registered buffer and should already be on the correct device
            // due to `TransformerTimeSeriesModel.to(device)` being called during model initialization. 
            // Therefore, it can be used directly here, assuming its device matches `x.device`. 
            // This avoids redundant copies and potential issues with buffer lifecycle.
            // Corrected to use get_buffer to retrieve the managed tensor.
            // Using null-forgiving operator since 'pe' is guaranteed to be registered.
            Tensor peOnDevice = this.get_buffer("pe")!; 

            // All temporary tensors created within this method should be managed by a DisposeScope.
            using (var innerScope = NewDisposeScope())
            {
                // Narrow `peOnDevice` to match the current sequence length of x. 
                // `narrow` typically returns a view. It will be disposed by `innerScope`.
                // Using null-forgiving operator since 'peOnDevice' is guaranteed non-null.
                Tensor slicedPe = torch.narrow(peOnDevice!, 1, 0, x.shape[1]);
                
                // Add positional encoding to input.
                // The result of `x + slicedPe` is a new Tensor and will be disposed by `innerScope`.
                Tensor xWithPositionalEncoding = x + slicedPe;
                
                // Apply dropout when the model is in training mode. 
                // MoveToOuterDisposeScope() ensures the final result tensor outlives this innerScope.
                return dropout(xWithPositionalEncoding, _dropout, this.training).MoveToOuterDisposeScope();
            }
        }
    }
}