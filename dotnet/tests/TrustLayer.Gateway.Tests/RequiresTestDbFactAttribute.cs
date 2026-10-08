namespace TrustLayer.Gateway.Tests;

// xUnit 2.9 has no Assert.Skip (that is xUnit.v3). The v2 pattern for a
// runtime-gated test is a FactAttribute that sets Skip at discovery —
// the constructor runs in the test process, so env reads work there.
public sealed class RequiresTestDbFactAttribute : FactAttribute
{
    public RequiresTestDbFactAttribute()
    {
        if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TRUSTLAYER_TEST_DB")))
        {
            Skip = "TRUSTLAYER_TEST_DB not set";
        }
    }
}
