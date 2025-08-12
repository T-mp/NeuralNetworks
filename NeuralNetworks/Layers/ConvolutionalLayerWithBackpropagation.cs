using Ivankarez.NeuralNetworks.Abstractions;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class ConvolutionalLayerWithBackpropagation : ConvolutionalLayer, IModelLayerWithBackpropagation
    {
        private float[]? lastInput;
        public ConvolutionalLayerWithBackpropagation(int filterSize, int stride, bool useBias, IInitializer kernelInitializer, IInitializer biasInitializer)
            : base(filterSize, stride, useBias, kernelInitializer, biasInitializer)
        { }

        public override float[] Update(float[] inputValues)
        {
            lastInput = inputValues; // Eingang merken für Backprop
            return base.Update(inputValues); // Original Forward 
        }

        // Backpropagation: OutputError zurückpropagieren, Gewichte updaten
        public float[] Backward(float[] outputError, float learningRate)
        {
            if(!IsBildet)
                throw new InvalidOperationException("Layer must be built before Backward can be called.");
            if (lastInput == null)
                throw new InvalidOperationException("Update must be called before Backward to set lastInput.");

            // delta = outputError, da keine Aktivierung verwendet wird
            var delta = outputError;
            var inputError = new float[lastInput.Length];

            // Jeder Output-Knoten
            for (int o = 0; o < OutputSize.TotalSize; o++)
            {
                float gradVal = delta[o];

                // Bias-Update nur falls verwendet
                if (UseBias)
                {
                    biases[o] -= learningRate * gradVal;
                }

                // Über alle Gewichte des (einen) Filters laufen
                for (int k = 0; k < FilterSize; k++)
                {
                    int inIndex = o * Stride + k;

                    // Prüfen ob innerhalb des Inputbereichs
                    if (inIndex >= 0 && inIndex < lastInput.Length)
                    {
                        // Gewicht anpassen
                        filter[k] -= learningRate * gradVal * lastInput[inIndex];

                        // Fehler für vorherigen Layer aufsummieren
                        inputError[inIndex] += filter[k] * gradVal;
                    }
                }
            }

            return inputError;
        }
    }
}
