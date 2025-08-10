dotnet restore
dotnet build --no-restore --configuration Release
dotnet test --no-build --verbosity normal
dotnet pack NeuralNetworks