using System.Reflection;

namespace TrustLayer.Gateway.Tests;

/// <summary>
/// Phase 1 placeholder. Deliberately not an empty [Fact]: that would pass even if
/// the project reference were misconfigured or the gateway stopped being an
/// application. This asserts the reference resolves and the assembly is runnable.
/// Phase 8 replaces it with real auth and ACL tests.
/// </summary>
public class GatewaySmokeTests
{
    [Fact]
    public void Gateway_assembly_resolves_and_is_an_application()
    {
        var assembly = Assembly.Load("TrustLayer.Gateway");

        Assert.Equal("TrustLayer.Gateway", assembly.GetName().Name);
        Assert.NotNull(assembly.EntryPoint);
    }
}
