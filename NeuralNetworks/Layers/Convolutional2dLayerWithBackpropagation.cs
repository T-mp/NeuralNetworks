using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.Utils;
using System;

namespace Ivankarez.NeuralNetworks.Layers;

public class Convolutional2dLayerWithBackpropagation : Convolutional2dLayer, IModelLayerWithBackpropagation
{
    // Optional: additional buffers as needed

    public Convolutional2dLayerWithBackpropagation(
        Size2D filterSize, Stride2D stride, bool useBias,
        IInitializer kernelInitializer, IInitializer biasInitializer
    ) : base(filterSize, stride, useBias, kernelInitializer, biasInitializer)
    { }

    public float[] Backward(float[] outputError, float learningRate)
    {
        if (!IsBildet)
        {
            throw new InvalidOperationException("Layer must be built before Backward can be called.");
        }

        // inputError-Array
        var inputError = new float[InputSize.TotalSize];
        // Gradient arrays for filter and bias
        var filterGradient = new float[filter.GetLength(0), filter.GetLength(1)];
        float[]? biasesGradient = UseBias ? new float[biases.Length] : null;

        // Für jeden Output-Knoten (wie im Forward)
        for (int nodeX = 0; nodeX < outputWidth; nodeX++)
        {
            for (int nodeY = 0; nodeY < outputHeight; nodeY++)
            {
                var nodeIndex = nodeX * outputHeight + nodeY;

                var delta = outputError[nodeIndex];

                // Filter-Gradi­enten und Backprop in Input-Error
                for (int fx = 0; fx < filter.GetLength(0); fx++)
                {
                    for (int fy = 0; fy < filter.GetLength(1); fy++)
                    {
                        var inputX = nodeX * Stride.Horizontal + fx;
                        var inputY = nodeY * Stride.Vertical + fy;

                        // [A] Gradient für diesen Filter-Wert aufsummieren
                        // inputX*InputSize.Width+inputY: Index in 1D-Input-Array
                        var inputIdx = inputX * InputSize.Width + inputY;
                        if (inputIdx >= 0 && inputIdx < nodeValues.Length)
                        {
                            filterGradient[fx, fy] += nodeValues[inputIdx] * delta;
                            // [B] Fehler ins Input-Error rückpropagieren
                            inputError[inputIdx] += filter[fx, fy] * delta;
                        }
                    }
                }

                // Bias-Gradient und -Update (falls aktiviert)
                if (UseBias && biasesGradient != null)
                {
                    biasesGradient[nodeIndex] += delta;
                }
            }
        }

        // **Update Filter**
        for (int fx = 0; fx < filter.GetLength(0); fx++)
        {
            for (int fy = 0; fy < filter.GetLength(1); fy++)
            {
                filter[fx, fy] -= learningRate * filterGradient[fx, fy];
            }
        }
        // **Update Bias**
        if (UseBias && biasesGradient != null)
        {
            for (int b = 0; b < biases.Length; b++)
            {
                biases[b] -= learningRate * biasesGradient[b];
            }
        }
        return inputError;
    }
}