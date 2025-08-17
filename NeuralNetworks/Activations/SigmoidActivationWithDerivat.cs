using Ivankarez.NeuralNetworks.Abstractions;
using System;

namespace Ivankarez.NeuralNetworks.Activations
{
    public class SigmoidActivationWithDerivat : SigmoidActivation, IActivationWithDerivat
    {
        public float Derivat(float input)
        {
            var sigmoid = 1.0f / (1.0f + (float)Math.Exp(-input));
            return sigmoid * (1f - sigmoid);
        }
    }
}
