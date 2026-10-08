using System.Net;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using TrustLayer.Gateway.Models;
using TrustLayer.Gateway.Services;

namespace TrustLayer.Gateway.Tests;

public sealed class FakeStore : IDocumentStore
{
    // carol (Everyone-only) must never see doc-hr (HR-only, internal)
    // or doc-con (HR-only, confidential). doc-pub is visible to all.
    public static readonly SearchHit PubHit = new(
        "c-pub-0", "doc-pub", "Public memo", "public content here",
        "text", "public", 0.1);

    public static readonly SearchHit HrHit = new(
        "c-hr-0", "doc-hr", "HR plan", "hr content here",
        "text", "internal", 0.2);

    public static readonly SearchHit ConHit = new(
        "c-con-0", "doc-con", "Secret plan", "secret content here",
        "text", "confidential", 0.3);

    public Task<IReadOnlyList<SearchHit>> SearchAsync(
        float[] vector, string[] roles, int topK, CancellationToken ct)
    {
        // The fake applies the same pre-filter the SQL does; the re-check
        // lives in DocumentStore (integration-tested), not here.
        var roleSet = roles.ToHashSet();
        var all = new[] { PubHit, HrHit, ConHit };
        var allowed = new Dictionary<string, string[]>
        {
            ["doc-pub"] = ["Everyone"],
            ["doc-hr"] = ["HR"],
            ["doc-con"] = ["HR"],
        };
        var labels = new Dictionary<string, string>
        {
            ["doc-pub"] = "public",
            ["doc-hr"] = "internal",
            ["doc-con"] = "confidential",
        };
        var hits = all
            .Where(h =>
            {
                var shared = allowed[h.DocId].Intersect(roleSet).ToArray();
                if (shared.Length == 0)
                {
                    return false;
                }

                return !string.Equals(labels[h.DocId], "confidential", StringComparison.Ordinal)
                    || shared.Any(r => r != "Everyone");
            })
            .Take(topK)
            .ToArray();
        return Task.FromResult<IReadOnlyList<SearchHit>>(hits);
    }

    public Task<DocumentResponse?> GetDocumentAsync(string docId, CancellationToken ct)
    {
        DocumentResponse? doc = docId switch
        {
            "doc-pub" => new("doc-pub", "Public memo", "text", "en", "public",
                [new(0, "public content here")]),
            "doc-hr" => new("doc-hr", "HR plan", "text", "en", "internal",
                [new(0, "hr content here")]),
            "doc-con" => new("doc-con", "Secret plan", "text", "en", "confidential",
                [new(0, "secret content here")]),
            _ => null,
        };
        return Task.FromResult(doc);
    }

    public Task<string[]?> GetAllowedRolesAsync(string docId, CancellationToken ct)
    {
        string[]? roles = docId switch
        {
            "doc-pub" => ["Everyone"],
            "doc-hr" => ["HR"],
            "doc-con" => ["HR"],
            _ => null,
        };
        return Task.FromResult(roles);
    }

    public Task<string> HealthAsync(CancellationToken ct) => Task.FromResult("ok");
}

public sealed class FakeModels : IModelService
{
    public Task<(float[] Vector, double LatencyMs)> EmbedQueryAsync(
        string text, string? requestId, CancellationToken ct) =>
        Task.FromResult((new float[768], 1.5));

    public Task<ServiceClassifyResponse> ClassifyAsync(
        string title, string text, string? requestId, CancellationToken ct) =>
        Task.FromResult(new ServiceClassifyResponse(
            "internal",
            new Dictionary<string, double>
            {
                ["public"] = 0.1,
                ["internal"] = 0.8,
                ["confidential"] = 0.1,
            },
            "lr-768cls", 2.5));

    public Task<ServiceAnswerResponse> AnswerAsync(
        string query, string[] passages, string? requestId, CancellationToken ct) =>
        Task.FromResult(new ServiceAnswerResponse($"Answer to '{query}' [1]", 3.5));

    public Task<string> HealthAsync(CancellationToken ct) => Task.FromResult("ok");
}

public class GatewayContractTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly WebApplicationFactory<Program> _factory;

    public GatewayContractTests(WebApplicationFactory<Program> factory)
    {
        _factory = factory.WithWebHostBuilder(builder =>
        {
            builder.UseEnvironment("Testing");
            builder.UseSetting("Jwt:Secret", "test-secret-32-chars-minimum-ok!");
            builder.ConfigureTestServices(services =>
            {
                services.AddSingleton<IDocumentStore, FakeStore>();
                services.AddSingleton<IModelService, FakeModels>();
            });
        });
    }

    private HttpClient Client() => _factory.CreateClient();

    private static async Task<string> LoginAsync(HttpClient client, string user)
    {
        var resp = await client.PostAsJsonAsync("/auth/login", new { username = user });
        resp.EnsureSuccessStatusCode();
        var body = await resp.Content.ReadFromJsonAsync<LoginResponse>();
        Assert.NotNull(body);
        return body.Token;
    }

    private static void Auth(HttpClient client, string token) =>
        client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", token);

    [Fact]
    public async Task Login_DemoUser_ReturnsTokenAndRoles()
    {
        var client = Client();
        var resp = await client.PostAsJsonAsync("/auth/login", new { username = "alice" });
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        var body = await resp.Content.ReadFromJsonAsync<LoginResponse>();
        Assert.NotNull(body);
        Assert.Equal("alice", body.Username);
        Assert.Equal(["HR", "Everyone"], body.Roles);
        Assert.NotEmpty(body.Token);
    }

    [Fact]
    public async Task Login_UnknownUser_Returns401()
    {
        var client = Client();
        var resp = await client.PostAsJsonAsync("/auth/login", new { username = "mallory" });
        Assert.Equal(HttpStatusCode.Unauthorized, resp.StatusCode);
    }

    [Fact]
    public async Task Ask_WithoutToken_Returns401()
    {
        var client = Client();
        var resp = await client.PostAsJsonAsync("/ask", new { query = "hello" });
        Assert.Equal(HttpStatusCode.Unauthorized, resp.StatusCode);
    }

    [Fact]
    public async Task Ask_TamperedToken_Returns401()
    {
        var client = Client();
        var token = await LoginAsync(client, "alice");
        Auth(client, token[..^4] + "XXXX");
        var resp = await client.PostAsJsonAsync("/ask", new { query = "hello" });
        Assert.Equal(HttpStatusCode.Unauthorized, resp.StatusCode);
    }

    [Fact]
    public async Task Ask_Fast_ReturnsResultsAndLatencyBreakdown()
    {
        var client = Client();
        Auth(client, await LoginAsync(client, "alice"));
        var resp = await client.PostAsJsonAsync("/ask", new { query = "plan", top_k = 3 });
        resp.EnsureSuccessStatusCode();
        var body = await resp.Content.ReadFromJsonAsync<AskResponse>();
        Assert.NotNull(body);
        Assert.Equal(3, body.Results.Length);
        Assert.Null(body.Answer);
        Assert.Empty(body.Citations);
        Assert.Equal(1.5, body.LatencyMs.Embed);
        Assert.True(body.LatencyMs.Total >= 0);
        Assert.True(resp.Headers.Contains("X-Request-Id"));
        var hit = body.Results[0];
        Assert.Equal("doc-pub", hit.DocId);
        Assert.Equal("c-pub-0", hit.ChunkId);
        Assert.InRange(hit.Score, 0.0, 1.0);
    }

    [Fact]
    public async Task Ask_FiltersInvisibleDocs_ThroughApi()
    {
        // carol (Everyone-only) must see doc-pub only — never doc-hr/conf.
        var client = Client();
        Auth(client, await LoginAsync(client, "carol"));
        var resp = await client.PostAsJsonAsync("/ask", new { query = "plan", top_k = 10 });
        resp.EnsureSuccessStatusCode();
        var body = await resp.Content.ReadFromJsonAsync<AskResponse>();
        Assert.NotNull(body);
        Assert.Single(body.Results);
        Assert.Equal("doc-pub", body.Results[0].DocId);
    }

    [Fact]
    public async Task Ask_Answer_ReturnsAnswerAndAlignedCitations()
    {
        var client = Client();
        Auth(client, await LoginAsync(client, "bob"));
        var resp = await client.PostAsJsonAsync("/ask", new { query = "budget", mode = "answer" });
        resp.EnsureSuccessStatusCode();
        var body = await resp.Content.ReadFromJsonAsync<AskResponse>();
        Assert.NotNull(body);
        Assert.NotNull(body.Answer);
        Assert.Contains("budget", body.Answer);
        Assert.Equal(body.Results.Length, body.Citations.Length);
        Assert.True(body.LatencyMs.Llm > 0);
        var returned = body.Results.Select(h => h.DocId).ToHashSet();
        Assert.All(body.Citations, c => Assert.Contains(c.DocId, returned));
    }

    [Theory]
    [InlineData("", "fast", 10, 768)]
    [InlineData("q", "summarize", 10, 768)]
    [InlineData("q", "fast", 0, 768)]
    [InlineData("q", "fast", 51, 768)]
    [InlineData("q", "fast", 10, 256)]
    public async Task Ask_InvalidInput_Returns400(
        string query, string mode, int topK, int dim)
    {
        var client = Client();
        Auth(client, await LoginAsync(client, "alice"));
        var resp = await client.PostAsJsonAsync(
            "/ask", new { query, mode, top_k = topK, dimension = dim });
        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
    }

    [Fact]
    public async Task Classify_Text_ReturnsLabelProbsMethod()
    {
        var client = Client();
        Auth(client, await LoginAsync(client, "alice"));
        var resp = await client.PostAsJsonAsync("/classify", new { text = "hello" });
        resp.EnsureSuccessStatusCode();
        var body = await resp.Content.ReadFromJsonAsync<ClassifyResponse>();
        Assert.NotNull(body);
        Assert.Equal("internal", body.Label);
        Assert.Equal("lr-768cls", body.Method);
        Assert.Null(body.DocId);
        Assert.Equal(3, body.Probs.Count);
    }

    [Fact]
    public async Task Classify_DocId_RespectsAcl()
    {
        var client = Client();
        Auth(client, await LoginAsync(client, "carol"));
        var ok = await client.PostAsJsonAsync("/classify", new { doc_id = "doc-pub" });
        Assert.Equal(HttpStatusCode.OK, ok.StatusCode);
        var hidden = await client.PostAsJsonAsync("/classify", new { doc_id = "doc-hr" });
        Assert.Equal(HttpStatusCode.NotFound, hidden.StatusCode);
        var missing = await client.PostAsJsonAsync("/classify", new { doc_id = "nope" });
        Assert.Equal(HttpStatusCode.NotFound, missing.StatusCode);
    }

    [Fact]
    public async Task Classify_BothOrNeither_Returns400()
    {
        var client = Client();
        Auth(client, await LoginAsync(client, "alice"));
        var both = await client.PostAsJsonAsync(
            "/classify", new { text = "x", doc_id = "doc-pub" });
        Assert.Equal(HttpStatusCode.BadRequest, both.StatusCode);
        var neither = await client.PostAsJsonAsync("/classify", new { });
        Assert.Equal(HttpStatusCode.BadRequest, neither.StatusCode);
    }

    [Fact]
    public async Task Documents_Visible_ReturnsDoc_InvisibleOrMissing_Returns404()
    {
        var client = Client();
        Auth(client, await LoginAsync(client, "carol"));
        var ok = await client.GetAsync("/documents/doc-pub");
        Assert.Equal(HttpStatusCode.OK, ok.StatusCode);
        var doc = await ok.Content.ReadFromJsonAsync<DocumentResponse>();
        Assert.NotNull(doc);
        Assert.Single(doc.Chunks);
        Assert.Equal(HttpStatusCode.NotFound, (await client.GetAsync("/documents/doc-con")).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await client.GetAsync("/documents/nope")).StatusCode);
    }

    [Fact]
    public async Task Health_IsUnauthenticated_AndReportsDeps()
    {
        var client = Client();
        var resp = await client.GetAsync("/health");
        resp.EnsureSuccessStatusCode();
        var body = await resp.Content.ReadFromJsonAsync<HealthResponse>();
        Assert.NotNull(body);
        Assert.Equal("ok", body.Status);
        Assert.Equal("ok", body.Db);
        Assert.Equal("ok", body.ModelService);
    }
}
