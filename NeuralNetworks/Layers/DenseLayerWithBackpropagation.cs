using Ivankarez.NeuralNetworks.Abstractions;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class DenseLayerWithBackpropagation : DenseLayer, IModelLayerWithBackpropagation
    {
        protected float[] Inputs = default!;

        public DenseLayerWithBackpropagation(int nodeCount, IActivationWithDerivat activation, bool useBias, IInitializer kernelInitializer, IInitializer biasInitializer) :
            base(nodeCount, activation, useBias, kernelInitializer, biasInitializer)
        {
        }

        override public float[] Update(float[] inputValues)
        {
            Inputs = inputValues ?? throw new ArgumentNullException(nameof(inputValues), "Input values cannot be null");
            return base.Update(inputValues);
        }

        public float[] Backward(float[] outputError, float learningRate)
        {
            var derivative = activation as IActivationWithDerivat
                ?? throw new InvalidOperationException("Activation function must implement IActivationWithRevert for backpropagation.");

            float[] inputError = new float[Inputs.Length];

            for (int j = 0; j < nodeValues.Length; j++)
            {
                float delta = outputError[j] * derivative.Derivat(nodeValues[j]);
                for (int i = 0; i < Inputs.Length; i++)
                {
                    inputError[i] += weights[i, j] * delta;
                    weights[i, j] -= learningRate * Inputs[i] * delta;  // Gewichte anpassen
                }
                if (useBias)
                {
                    biases[j] -= learningRate * delta;  // Bias anpassen
                }
            }
            return inputError; ;
        }
    }
}
