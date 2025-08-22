using Ivankarez.NeuralNetworks.Abstractions;
using System;

namespace Ivankarez.NeuralNetworks
{
    public class LayeredNetworkModelWithBackpropagation : LayeredNetworkModel
    {
        public LayeredNetworkModelWithBackpropagation(ISize inputs, params IModelLayerWithBackpropagation[] layers)
            : base(inputs, layers)
        {}

        public float[] Backward(float[] outputError, float learningRate)
        {
            if (Layers.Count == 0) throw new InvalidOperationException("No layers to backpropagate through");
            var layerErrors = outputError;
            for (int i = Layers.Count - 1; i >= 0; i--)
            {
                var layer = (IModelLayerWithBackpropagation)Layers[i];
                layerErrors = layer.Backward(layerErrors, learningRate);
            }
            return layerErrors;
        }
    }
}
