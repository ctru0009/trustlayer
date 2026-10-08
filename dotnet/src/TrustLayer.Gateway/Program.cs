using System.Diagnostics;
using System.Security.Claims;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Authorization;
using TrustLayer.Gateway.Auth;
using TrustLayer.Gateway.Models;
using TrustLayer.Gateway.Services;

var builder = WebApplication.CreateBuilder(args);

var secret = builder.Configuration["Jwt:Secret"];
if (string.IsNullOrWhiteSpace(secret))
{
    if (builder.Environment.IsDevelopment())
    {
        secret = "dev-only-change-me-32-chars-minimum";
    }
    else
    {
        throw new InvalidOperationException("Jwt:Secret must be configured outside Development.");
    }
}

var tokens = new TokenService(secret);
builder.Services.AddSingleton(tokens);
builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
    .AddJwtBearer(options => options.TokenValidationParameters = tokens.ValidationParameters());
builder.Services.AddAuthorization();
builder.Services.AddHttpClient<IModelService, ModelServiceClient>(client =>
    client.BaseAddress = new Uri(
        builder.Configuration["ModelService:BaseUrl"] ?? "http://localhost:8000"));
builder.Services.AddSingleton<IDocumentStore>(_ => new DocumentStore(
    builder.Configuration.GetConnectionString("TrustLayer")
    ?? "Host=localhost;Port=5432;Database=trustlayer;Username=trustlayer;Password=trustlayer_local"));

var app = builder.Build();

// Request IDs: accept X-Request-Id or generate; echo it back and scope logs.
app.Use(async (context, next) =>
{
    var id = context.Request.Headers["X-Request-Id"].FirstOrDefault();
    if (string.IsNullOrEmpty(id))
    {
        id = Guid.NewGuid().ToString("N");
    }

    context.Response.Headers["X-Request-Id"] = id;
    using (app.Logger.BeginScope(new Dictionary<string, object?> { ["request_id"] = id }))
    {
        await next(context).ConfigureAwait(false);
    }
});

app.UseAuthentication();
app.UseAuthorization();

static string[] RolesOf(ClaimsPrincipal user) =>
    user.FindAll(ClaimTypes.Role).Select(c => c.Value).ToArray();

app.MapPost("/auth/login", (LoginRequest body) =>
{
    if (string.IsNullOrWhiteSpace(body.Username)
        || !Acl.Users.TryGetValue(body.Username.Trim(), out var roles))
    {
        return Results.Unauthorized();
    }

    var username = body.Username.Trim();
    return Results.Ok(new LoginResponse(tokens.Issue(username, roles), username, roles));
});

app.MapGet("/health", async (IDocumentStore store, IModelService models, CancellationToken ct) =>
{
    var db = await store.HealthAsync(ct).ConfigureAwait(false);
    var svc = await models.HealthAsync(ct).ConfigureAwait(false);
    return Results.Ok(new HealthResponse("ok", db, svc));
});

app.MapPost("/ask", [Authorize] async (
    AskRequest body,
    ClaimsPrincipal user,
    IDocumentStore store,
    IModelService models,
    HttpContext context,
    CancellationToken ct) =>
{
    if (string.IsNullOrWhiteSpace(body.Query))
    {
        return Results.BadRequest(new { detail = "query must not be empty" });
    }

    if (!string.Equals(body.Mode, "fast", StringComparison.Ordinal)
        && !string.Equals(body.Mode, "answer", StringComparison.Ordinal))
    {
        return Results.BadRequest(new { detail = "mode must be fast or answer" });
    }

    if (body.TopK is < 1 or > 50)
    {
        return Results.BadRequest(new { detail = "top_k must be 1..50" });
    }

    // Spec section 6: queries and documents must use the same dimension.
    // chunks.embedding is vector(768); anything else scores garbage.
    if (body.Dimension != 768)
    {
        return Results.BadRequest(new { detail = "dimension must be 768" });
    }

    var total = Stopwatch.StartNew();
    var roles = RolesOf(user);
    var requestId = context.Response.Headers["X-Request-Id"].FirstOrDefault();
    (float[] Vector, double LatencyMs) embedded;
    try
    {
        embedded = await models.EmbedQueryAsync(body.Query, requestId, ct).ConfigureAwait(false);
    }
    catch (HttpRequestException ex)
    {
        return Results.Problem("model service unavailable: " + ex.Message, statusCode: 503);
    }

    if (embedded.Vector.Length != 768)
    {
        return Results.Problem("model service returned wrong dimension", statusCode: 502);
    }

    var searchWatch = Stopwatch.StartNew();
    IReadOnlyList<SearchHit> hits;
    try
    {
        hits = await store.SearchAsync(embedded.Vector, roles, body.TopK, ct).ConfigureAwait(false);
    }
    catch (AclDriftException ex)
    {
        GatewayLog.AclDrift(app.Logger, ex);
        return Results.Problem("internal error", statusCode: 500);
    }

    var searchMs = searchWatch.Elapsed.TotalMilliseconds;
    // Cosine distance from pgvector (<=> on normalized vectors, 0..2);
    // score is similarity so higher ranks first.
    var results = hits.Select(h => new AskHit(
        h.DocId, h.ChunkId, h.Title, h.Snippet, h.Modality, 1.0 - h.Distance, h.Label)).ToArray();

    if (string.Equals(body.Mode, "fast", StringComparison.Ordinal))
    {
        return Results.Ok(new AskResponse(
            results, null, [],
            new LatencyBreakdown(embedded.LatencyMs, searchMs, 0, total.Elapsed.TotalMilliseconds)));
    }

    string[] passages = hits.Select(h =>
        h.Snippet.Length > 1500 ? h.Snippet[..1500] : h.Snippet).ToArray();
    ServiceAnswerResponse answered;
    try
    {
        answered = await models.AnswerAsync(body.Query, passages, requestId, ct).ConfigureAwait(false);
    }
    catch (HttpRequestException ex)
    {
        return Results.Problem("model service unavailable: " + ex.Message, statusCode: 503);
    }

    var citations = hits.Select(h => new Citation(h.DocId, h.ChunkId, h.Title)).ToArray();
    return Results.Ok(new AskResponse(
        results, answered.Answer, citations,
        new LatencyBreakdown(
            embedded.LatencyMs, searchMs, answered.LatencyMs, total.Elapsed.TotalMilliseconds)));
});

app.MapPost("/classify", [Authorize] async (
    ClassifyRequest body,
    ClaimsPrincipal user,
    IDocumentStore store,
    IModelService models,
    HttpContext context,
    CancellationToken ct) =>
{
    var hasText = !string.IsNullOrWhiteSpace(body.Text);
    var hasDoc = !string.IsNullOrWhiteSpace(body.DocId);
    if (hasText == hasDoc)
    {
        return Results.BadRequest(new { detail = "exactly one of text or doc_id" });
    }

    string title;
    string text;
    string? docId = null;
    if (hasText)
    {
        title = body.Title ?? string.Empty;
        text = body.Text!;
    }
    else
    {
        docId = body.DocId!.Trim();
        var candidate = await store.GetDocumentAsync(docId, ct).ConfigureAwait(false);
        var candidateRoles = await store.GetAllowedRolesAsync(docId, ct).ConfigureAwait(false);
        // Missing OR invisible → identical 404 (spec section 8.2: errors
        // never reveal whether a restricted document exists).
        if (candidate is null || candidateRoles is null
            || !Acl.Visible(candidate.Label, candidateRoles, RolesOf(user)))
        {
            return Results.NotFound();
        }

        title = candidate.Title;
        text = string.Join("\n", candidate.Chunks.Select(c => c.Text));
        if (text.Length > 4000)
        {
            text = text[..4000];
        }
    }

    var requestId = context.Response.Headers["X-Request-Id"].FirstOrDefault();
    ServiceClassifyResponse classified;
    try
    {
        classified = await models.ClassifyAsync(title, text, requestId, ct).ConfigureAwait(false);
    }
    catch (HttpRequestException ex)
    {
        return Results.Problem("model service unavailable: " + ex.Message, statusCode: 503);
    }

    return Results.Ok(new ClassifyResponse(
        classified.Label, classified.Probs, classified.Method, docId));
});

app.MapGet("/documents/{id}", [Authorize] async (
    string id,
    ClaimsPrincipal user,
    IDocumentStore store,
    CancellationToken ct) =>
{
    var doc = await store.GetDocumentAsync(id, ct).ConfigureAwait(false);
    var allowed = await store.GetAllowedRolesAsync(id, ct).ConfigureAwait(false);
    if (doc is null || allowed is null || !Acl.Visible(doc.Label, allowed, RolesOf(user)))
    {
        return Results.NotFound();
    }

    return Results.Ok(doc);
});

app.Run();

// Partial Program class for WebApplicationFactory entry-point discovery.
public partial class Program
{
}

internal static partial class GatewayLog
{
    [LoggerMessage(Level = LogLevel.Critical, Message = "ACL drift — failing loud, never leaking")]
    internal static partial void AclDrift(ILogger logger, Exception ex);
}
