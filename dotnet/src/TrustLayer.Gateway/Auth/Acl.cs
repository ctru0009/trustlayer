// Access-control predicate: who may see which document.
//
// Spec section 8.1 (port of trustlayer.retrieve.acl): a user may see a
// document if one of their roles is in allowed_roles; confidential
// documents additionally require an explicit (non-Everyone) role grant.

namespace TrustLayer.Gateway.Auth;

public static class Acl
{
    public static readonly IReadOnlyDictionary<string, string[]> Users =
        new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase)
        {
            ["alice"] = ["HR", "Everyone"],
            ["bob"] = ["Finance", "Everyone"],
            ["carol"] = ["Everyone"],
            ["admin"] = ["HR", "Finance", "Everyone", "Admin"],
        };

    public static bool Visible(string label, IEnumerable<string> allowedRoles, IEnumerable<string> userRoles)
    {
        var shared = allowedRoles.Intersect(userRoles).ToHashSet(StringComparer.Ordinal);
        if (shared.Count == 0)
        {
            return false;
        }

        if (string.Equals(label, "confidential", StringComparison.Ordinal))
        {
            shared.Remove("Everyone");
            return shared.Count > 0;
        }

        return true;
    }

    public static string[] NonEveryoneRoles(IEnumerable<string> userRoles) =>
        userRoles.Where(r => !string.Equals(r, "Everyone", StringComparison.Ordinal)).ToArray();
}
