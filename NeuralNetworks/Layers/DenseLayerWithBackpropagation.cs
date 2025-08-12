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
            if (!IsBildet)
                throw new InvalidOperationException("Layer must be built before Backward can be called.");
            if (Inputs == null)
                throw new InvalidOperationException("Update must be called before Backward to set Inputs.");

            var derivative = activation as IActivationWithDerivat
                ?? throw new InvalidOperationException("Activation function must implement IActivationWithRevert for backpropagation.");

            float[] inputError = new float[Inputs.Length];

            for (int o = 0; o < nodeValues.Length; o++)
            {
                float delta = outputError[o] * derivative.Derivat(nodeValues[o]);
                for (int i = 0; i < Inputs.Length; i++)
                {
                    inputError[i] += weights[o, i] * delta;
                    weights[o, i] -= learningRate * Inputs[i] * delta;  // Gewichte anpassen
                }
                if (useBias)
                {
                    biases[o] -= learningRate * delta;  // Bias anpassen
                }
            }
            return inputError; ;
        }
    }
}
