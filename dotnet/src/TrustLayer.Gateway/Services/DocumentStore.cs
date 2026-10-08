using System.Globalization;
using Npgsql;
using TrustLayer.Gateway.Auth;
using TrustLayer.Gateway.Models;
namespace TrustLayer.Gateway.Services;

public sealed class DocumentStore(string connectionString) : IDocumentStore
{
    // Port of trustlayer.retrieve.search: the permission predicate lives in
    // the SQL itself (pre-filter), then every hit is re-checked with Acl.
    // strict_order: filtered HNSW can return fewer than LIMIT rows.
    private const string SearchSql = """
        SELECT c.id, c.doc_id, d.title, c.text, d.modality, d.label,
               c.embedding <=> $1::vector AS dist
        FROM chunks c JOIN documents d ON d.id = c.doc_id
        WHERE d.allowed_roles && $2::text[]
          AND (d.label != 'confidential' OR d.allowed_roles && $3::text[])
        ORDER BY c.embedding <=> $1::vector LIMIT $4
        """;

    public async Task<IReadOnlyList<SearchHit>> SearchAsync(
        float[] vector, string[] roles, int topK, CancellationToken ct)
    {
        var vec = "[" + string.Join(",", vector.Select(v => v.ToString("R", CultureInfo.InvariantCulture))) + "]";
        await using var conn = new NpgsqlConnection(connectionString);
        await conn.OpenAsync(ct).ConfigureAwait(false);
        await using (var cmd = new NpgsqlCommand("SET hnsw.iterative_scan = 'strict_order'", conn))
        {
            await cmd.ExecuteNonQueryAsync(ct).ConfigureAwait(false);
        }

        await using var search = new NpgsqlCommand(SearchSql, conn);
        search.Parameters.AddWithValue(vec);
        search.Parameters.AddWithValue(roles);
        search.Parameters.AddWithValue(Acl.NonEveryoneRoles(roles));
        search.Parameters.AddWithValue(topK);
        var hits = new List<SearchHit>();
        await using (var reader = await search.ExecuteReaderAsync(ct).ConfigureAwait(false))
        {
            while (await reader.ReadAsync(ct).ConfigureAwait(false))
            {
                var text = reader.GetString(3);
                hits.Add(new SearchHit(
                    reader.GetString(0),
                    reader.GetString(1),
                    reader.GetString(2),
                    text.Length > 200 ? text[..200] : text,
                    reader.GetString(4),
                    reader.GetString(5),
                    reader.GetDouble(6)));
            }
        }

        // Defence in depth: re-check every hit against Acl.Visible. Drift
        // between the SQL predicate and this check is a bug — throw, never
        // silently drop (dropping would hide a leak in the other direction).
        await using var recheck = new NpgsqlCommand(
            "SELECT id, label, allowed_roles FROM documents WHERE id = ANY($1)", conn);
        recheck.Parameters.AddWithValue(hits.Select(h => h.DocId).ToArray());
        var granted = new Dictionary<string, (string Label, string[] Roles)>(StringComparer.Ordinal);
        await using (var reader = await recheck.ExecuteReaderAsync(ct).ConfigureAwait(false))
        {
            while (await reader.ReadAsync(ct).ConfigureAwait(false))
            {
                granted[reader.GetString(0)] =
                    (reader.GetString(1), (string[])reader.GetValue(2));
            }
        }

        foreach (var hit in hits)
        {
            var (label, allowed) = granted[hit.DocId];
            if (!Acl.Visible(label, allowed, roles))
            {
                throw new AclDriftException(hit.DocId);
            }
        }

        return hits;
    }

    public async Task<DocumentResponse?> GetDocumentAsync(string docId, CancellationToken ct)
    {
        await using var conn = new NpgsqlConnection(connectionString);
        await conn.OpenAsync(ct).ConfigureAwait(false);
        await using var doc = new NpgsqlCommand(
            "SELECT id, title, modality, lang, label FROM documents WHERE id = $1", conn);
        doc.Parameters.AddWithValue(docId);
        await using var reader = await doc.ExecuteReaderAsync(ct).ConfigureAwait(false);
        if (!await reader.ReadAsync(ct).ConfigureAwait(false))
        {
            return null;
        }

        var response = new DocumentResponse(
            reader.GetString(0), reader.GetString(1), reader.GetString(2),
            reader.GetString(3), reader.GetString(4), []);
        await reader.CloseAsync().ConfigureAwait(false);

        await using var chunks = new NpgsqlCommand(
            "SELECT ord, text FROM chunks WHERE doc_id = $1 ORDER BY ord", conn);
        chunks.Parameters.AddWithValue(docId);
        var list = new List<DocumentChunk>();
        await using (var creader = await chunks.ExecuteReaderAsync(ct).ConfigureAwait(false))
        {
            while (await creader.ReadAsync(ct).ConfigureAwait(false))
            {
                list.Add(new DocumentChunk(creader.GetInt32(0), creader.GetString(1)));
            }
        }

        return response with { Chunks = list.ToArray() };
    }

    public async Task<string[]?> GetAllowedRolesAsync(string docId, CancellationToken ct)
    {
        await using var conn = new NpgsqlConnection(connectionString);
        await conn.OpenAsync(ct).ConfigureAwait(false);
        await using var cmd = new NpgsqlCommand(
            "SELECT allowed_roles FROM documents WHERE id = $1", conn);
        cmd.Parameters.AddWithValue(docId);
        var value = await cmd.ExecuteScalarAsync(ct).ConfigureAwait(false);
        return value is string[] roles ? roles : null;
    }

    public async Task<string> HealthAsync(CancellationToken ct)
    {
        try
        {
            await using var conn = new NpgsqlConnection(connectionString);
            await conn.OpenAsync(ct).ConfigureAwait(false);
            await using var cmd = new NpgsqlCommand("SELECT 1", conn);
            await cmd.ExecuteScalarAsync(ct).ConfigureAwait(false);
            return "ok";
        }
        catch (NpgsqlException)
        {
            return "down";
        }
        catch (TaskCanceledException)
        {
            return "down";
        }
    }
}
