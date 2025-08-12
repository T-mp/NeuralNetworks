using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.RandomGeneration;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class DropoutLayerWithBackpropagation : DropoutLayer, IModelLayerWithBackpropagation
    {
        internal bool[] mask = [];
        public DropoutLayerWithBackpropagation(float dropoutRate, IRandomProvider randomProvider)
            : base(dropoutRate, randomProvider)
        { }

        public override void Build(ISize inputSize)
        {
            base.Build(inputSize);
            mask = new bool[OutputSize.TotalSize];
        }

        public override float[] Update(float[] inputValues)
        {
            if (!IsBildet)
                throw new InvalidOperationException("Layer must be built before Update can be called.");

            for (int i = 0; i < OutputSize.TotalSize; i++)
            {
                mask[i] = RandomProvider.NextBool(DropoutRate);
                nodeValues[i] = mask[i] ? inputValues[i] : 0;
            }

            return nodeValues;
        }
        public float[] Backward(float[] error, float learningRate)
        {
            if (!IsBildet)
                throw new InvalidOperationException("Layer must be built before Backward can be called.");

            float[] inputError = new float[OutputSize.TotalSize];
            for (int i = 0; i < OutputSize.TotalSize; i++)
            {
                inputError[i] = mask[i] ? error[i] : 0;
            }

            return inputError;
        }
    }
}
