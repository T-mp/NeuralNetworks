namespace Ivankarez.NeuralNetworks.Abstractions
{
    public interface IActivationWithDerivat:IActivation
    {
        public float Derivat(float input);
    }
}
