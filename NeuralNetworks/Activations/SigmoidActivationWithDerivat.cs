using Ivankarez.NeuralNetworks.Abstractions;

namespace Ivankarez.NeuralNetworks.Activations
{
    public class SigmoidActivationWithDerivat : SigmoidActivation, IActivationWithDerivat
    {
        public float Derivat(float input)
        {
            return input * (1f - input);
        }
    }
}
