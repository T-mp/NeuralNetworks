using Ivankarez.NeuralNetworks.Abstractions;

namespace Ivankarez.NeuralNetworks.Activations
{
    public class ReluActivationWithDerivat : ReluActivation, IActivationWithDerivat
    {
        public float Derivat(float input)
        {
            return input > 0f 
                ? 1f 
                : 0f;
        }
    }
}
