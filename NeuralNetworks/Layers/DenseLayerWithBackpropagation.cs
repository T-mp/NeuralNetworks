using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class DenseLayerWithBackpropagation : DenseLayer, IModelLayerWithBackpropagation
    {
        public NamedVectors<float> BackpropagationState { get; } = new NamedVectors<float>();
        protected override void BackpropagationStateSet(string name, int index, float value)
        {
            BackpropagationState.Get1dVector(name)[index] = value;
        }

        public DenseLayerWithBackpropagation(int nodeCount, IActivationWithDerivat activation, bool useBias, IInitializer kernelInitializer, IInitializer biasInitializer) :
            base(nodeCount, activation, useBias, kernelInitializer, biasInitializer)
        {
        }

        override public float[] Update(float[] inputValues)
        {
            BackpropagationState.Clear();
            BackpropagationState.Add("inputs", inputValues ?? throw new ArgumentNullException(nameof(inputValues), "Input values cannot be null"));
            BackpropagationState.Add("lastPreAct", new float[OutputSize.TotalSize]);
            return base.Update(inputValues);
        }

        public float[] Backward(float[] outputError, float learningRate)
        {
            if (!IsBildet)
                throw new InvalidOperationException("Layer must be built before Backward can be called.");
            if (!BackpropagationState.ContainsKey1D("inputs"))
                throw new InvalidOperationException("Update must be called before Backward to set Inputs.");

            var inputs = BackpropagationState.Get1dVector("inputs");
            var lastPreAct = BackpropagationState.Get1dVector("lastPreAct");
            var derivative = activation as IActivationWithDerivat
                ?? throw new InvalidOperationException("Activation function must implement IActivationWithRevert for backpropagation.");

            float[] inputError = new float[inputs.Length];

            for (int o = 0; o < nodeValues.Length; o++)
            {
                float delta = outputError[o] * derivative.Derivat(lastPreAct[o]);
                for (int i = 0; i < inputs.Length; i++)
                {
                    inputError[i] += weights[o, i] * delta;
                    weights[o, i] -= learningRate * inputs[i] * delta;  // Gewichte anpassen
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
