using System.Text.Json.Serialization;

namespace TrustLayer.Gateway.Models;

public sealed record LoginRequest(string Username);

public sealed record LoginResponse(string Token, string Username, string[] Roles);

public sealed record HealthResponse(string Status, string Db, string ModelService);

public sealed record AskRequest(
    string Query,
    string Mode = "fast",
    [property: JsonPropertyName("top_k")] int TopK = 10,
    int Dimension = 768);

public sealed record AskHit(
    [property: JsonPropertyName("doc_id")] string DocId,
    [property: JsonPropertyName("chunk_id")] string ChunkId,
    string Title,
    string Snippet,
    string Modality,
    double Score,
    string Label);

public sealed record Citation(
    [property: JsonPropertyName("doc_id")] string DocId,
    [property: JsonPropertyName("chunk_id")] string ChunkId,
    string Title);

public sealed record LatencyBreakdown(double Embed, double Search, double Llm, double Total);

public sealed record AskResponse(
    AskHit[] Results,
    string? Answer,
    Citation[] Citations,
    [property: JsonPropertyName("latency_ms")] LatencyBreakdown LatencyMs);

public sealed record ClassifyRequest(
    string? Text,
    string? Title,
    [property: JsonPropertyName("doc_id")] string? DocId);

public sealed record ClassifyResponse(
    string Label,
    Dictionary<string, double> Probs,
    string Method,
    [property: JsonPropertyName("doc_id")] string? DocId);

public sealed record DocumentChunk(int Ord, string Text);

public sealed record DocumentResponse(
    string Id,
    string Title,
    string Modality,
    string Lang,
    string Label,
    DocumentChunk[] Chunks);

public sealed record ServiceEmbedResponse(
    [property: JsonPropertyName("vectors")] float[][] Vectors,
    [property: JsonPropertyName("dim")] int Dim,
    [property: JsonPropertyName("latency_ms")] double LatencyMs);

public sealed record ServiceClassifyResponse(
    [property: JsonPropertyName("label")] string Label,
    [property: JsonPropertyName("probs")] Dictionary<string, double> Probs,
    [property: JsonPropertyName("method")] string Method,
    [property: JsonPropertyName("latency_ms")] double LatencyMs);

public sealed record ServiceAnswerResponse(
    [property: JsonPropertyName("answer")] string Answer,
    [property: JsonPropertyName("latency_ms")] double LatencyMs);
