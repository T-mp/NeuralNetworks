using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.RandomGeneration;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class DropoutLayerWithBackpropagation : DropoutLayer, IModelLayerWithBackpropagation
    {
        private bool[] mask = [];
        public DropoutLayerWithBackpropagation(float dropoutRate, IRandomProvider randomProvider)
            : base(dropoutRate, randomProvider)
        {}
        public override float[] Update(float[] inputValues)
        {
            for (int i = 0; i < OutputSize.TotalSize; i++)
            {
                mask[i] = RandomProvider.NextBool(DropoutRate);
                nodeValues[i] = mask[i] ? inputValues[i] : 0;
            }

            return nodeValues;
        }
        public float[] Backward(float[] error, float learningRate)
        {
            float[] inputError = new float[OutputSize.TotalSize];
            for (int i = 0; i < OutputSize.TotalSize; i++)
            {
                inputError[i] = mask[i] ? error[i] : 0;
            }

            return inputError;
        }
    }
}
