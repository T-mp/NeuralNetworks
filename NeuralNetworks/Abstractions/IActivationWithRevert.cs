namespace Ivankarez.NeuralNetworks.Abstractions
{
    public interface IActivationWithRevert:IActivation
    {
        public float Revert(float input);
    }
}
