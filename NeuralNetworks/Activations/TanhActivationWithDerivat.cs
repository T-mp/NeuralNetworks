using Ivankarez.NeuralNetworks.Abstractions;
using System;

namespace Ivankarez.NeuralNetworks.Activations
{
    public class TanhActivationWithDerivat : TanhActivation, IActivationWithDerivat
    {
        public float Derivat(float input)
        {
            return 1f - MathF.Tanh(input) * MathF.Tanh(input);
        }
    }
}
