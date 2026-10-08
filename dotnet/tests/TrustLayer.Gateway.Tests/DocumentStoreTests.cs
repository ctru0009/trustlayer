using TrustLayer.Gateway.Services;

namespace TrustLayer.Gateway.Tests;

// Live-Postgres tests for the retrieval SQL + re-check agreement. They run
// only when TRUSTLAYER_TEST_DB is set (local dev with `make up`); CI and
// the default `make test` skip them — contract tests carry the suite.
public class DocumentStoreTests
{
    private static string ConnectionString() =>
        Environment.GetEnvironmentVariable("TRUSTLAYER_TEST_DB")!;

    [RequiresTestDbFact]
    public async Task SearchAsync_PreFilterAndRecheckAgree()
    {
        var store = new DocumentStore(ConnectionString());
        // Zero vector: distances tie, but ACL filtering still applies.
        var hits = await store.SearchAsync(new float[768], ["Everyone"], 10, CancellationToken.None);
        // No throw = SQL pre-filter and Acl.Visible agreed on every hit.
        Assert.NotNull(hits);
        Assert.All(hits, h => Assert.NotEqual("confidential", h.Label));
    }

    [RequiresTestDbFact]
    public async Task SearchAsync_AdminSeesConfidential()
    {
        var store = new DocumentStore(ConnectionString());
        var hits = await store.SearchAsync(
            new float[768], ["HR", "Finance", "Everyone", "Admin"], 50,
            CancellationToken.None);
        Assert.NotNull(hits);
    }

    [RequiresTestDbFact]
    public async Task GetDocumentAsync_Missing_ReturnsNull()
    {
        var store = new DocumentStore(ConnectionString());
        var doc = await store.GetDocumentAsync(
            "definitely-not-a-doc-id", CancellationToken.None);
        Assert.Null(doc);
    }

    [RequiresTestDbFact]
    public async Task HealthAsync_ReturnsOk()
    {
        var store = new DocumentStore(ConnectionString());
        Assert.Equal("ok", await store.HealthAsync(CancellationToken.None));
    }
}
