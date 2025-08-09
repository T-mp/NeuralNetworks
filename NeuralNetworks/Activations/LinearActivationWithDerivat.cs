using Ivankarez.NeuralNetworks.Abstractions;

namespace Ivankarez.NeuralNetworks.Activations
{
    public class LinearActivationWithDerivat : LinearActivation, IActivationWithDerivat
    {
        public float Derivat(float input) => 1.0f;
    }
}
