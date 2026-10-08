using System.Net.Http.Json;
using TrustLayer.Gateway.Models;

namespace TrustLayer.Gateway.Services;

public sealed class AclDriftException(string docId)
    : InvalidOperationException($"ACL re-check failed: {docId}");

public sealed record SearchHit(
    string ChunkId,
    string DocId,
    string Title,
    string Snippet,
    string Modality,
    string Label,
    double Distance);

public interface IDocumentStore
{
    Task<IReadOnlyList<SearchHit>> SearchAsync(
        float[] vector, string[] roles, int topK, CancellationToken ct);

    Task<DocumentResponse?> GetDocumentAsync(string docId, CancellationToken ct);

    Task<string[]?> GetAllowedRolesAsync(string docId, CancellationToken ct);

    Task<string> HealthAsync(CancellationToken ct);
}

public interface IModelService
{
    Task<(float[] Vector, double LatencyMs)> EmbedQueryAsync(
        string text, string? requestId, CancellationToken ct);

    Task<ServiceClassifyResponse> ClassifyAsync(
        string title, string text, string? requestId, CancellationToken ct);

    Task<ServiceAnswerResponse> AnswerAsync(
        string query, string[] passages, string? requestId, CancellationToken ct);

    Task<string> HealthAsync(CancellationToken ct);
}

public sealed class ModelServiceClient(HttpClient http) : IModelService
{
    public async Task<(float[] Vector, double LatencyMs)> EmbedQueryAsync(
        string text, string? requestId, CancellationToken ct)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, "/embed")
        {
            Content = JsonContent.Create(new { texts = new[] { text }, kind = "query", dim = 768 }),
        };
        AddRequestId(request, requestId);
        using var response = await http.SendAsync(request, ct).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        var body = await response.Content.ReadFromJsonAsync<ServiceEmbedResponse>(ct)
            .ConfigureAwait(false);
        ArgumentNullException.ThrowIfNull(body);
        return (body.Vectors[0], body.LatencyMs);
    }

    public async Task<ServiceClassifyResponse> ClassifyAsync(
        string title, string text, string? requestId, CancellationToken ct)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, "/classify")
        {
            Content = JsonContent.Create(new { title, text }),
        };
        AddRequestId(request, requestId);
        using var response = await http.SendAsync(request, ct).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        var body = await response.Content.ReadFromJsonAsync<ServiceClassifyResponse>(ct)
            .ConfigureAwait(false);
        ArgumentNullException.ThrowIfNull(body);
        return body;
    }

    public async Task<ServiceAnswerResponse> AnswerAsync(
        string query, string[] passages, string? requestId, CancellationToken ct)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, "/answer")
        {
            Content = JsonContent.Create(new { query, passages }),
        };
        AddRequestId(request, requestId);
        using var response = await http.SendAsync(request, ct).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        var body = await response.Content.ReadFromJsonAsync<ServiceAnswerResponse>(ct)
            .ConfigureAwait(false);
        ArgumentNullException.ThrowIfNull(body);
        return body;
    }

    public async Task<string> HealthAsync(CancellationToken ct)
    {
        try
        {
            using var response = await http.GetAsync("/health", ct).ConfigureAwait(false);
            return response.IsSuccessStatusCode ? "ok" : "down";
        }
        catch (HttpRequestException)
        {
            return "down";
        }
        catch (TaskCanceledException)
        {
            return "down";
        }
    }

    private static void AddRequestId(HttpRequestMessage request, string? requestId)
    {
        if (!string.IsNullOrEmpty(requestId))
        {
            request.Headers.Add("X-Request-Id", requestId);
        }
    }
}
