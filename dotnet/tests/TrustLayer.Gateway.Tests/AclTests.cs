using TrustLayer.Gateway.Auth;

namespace TrustLayer.Gateway.Tests;

public class AclTests
{
    [Theory]
    // Public: any shared role suffices.
    [InlineData("public", new[] { "Everyone" }, new[] { "Everyone" }, true)]
    [InlineData("public", new[] { "Everyone" }, new[] { "HR", "Everyone" }, true)]
    [InlineData("public", new[] { "HR" }, new[] { "Finance", "Everyone" }, false)]
    // Internal: any shared role suffices.
    [InlineData("internal", new[] { "Everyone", "HR" }, new[] { "Everyone" }, true)]
    [InlineData("internal", new[] { "Everyone", "HR" }, new[] { "Finance", "Everyone" }, true)]
    [InlineData("internal", new[] { "HR" }, new[] { "Finance", "Everyone" }, false)]
    // Confidential: needs an explicit (non-Everyone) shared role.
    [InlineData("confidential", new[] { "HR" }, new[] { "HR", "Everyone" }, true)]
    [InlineData("confidential", new[] { "HR" }, new[] { "Everyone" }, false)]
    [InlineData("confidential", new[] { "Everyone" }, new[] { "HR", "Everyone" }, false)]
    [InlineData("confidential", new[] { "Finance" }, new[] { "HR", "Everyone" }, false)]
    [InlineData("confidential", new[] { "HR", "Finance" }, new[] { "Finance", "Everyone" }, true)]
    public void Visible_MatchesSpec(
        string label, string[] allowed, string[] roles, bool expected)
    {
        Assert.Equal(expected, Acl.Visible(label, allowed, roles));
    }

    [Fact]
    public void Users_HasFourDemoUsers()
    {
        Assert.Equal(
            ["admin", "alice", "bob", "carol"],
            Acl.Users.Keys.OrderBy(k => k).ToArray());
        Assert.Equal(["HR", "Everyone"], Acl.Users["alice"]);
        Assert.Equal(["Finance", "Everyone"], Acl.Users["bob"]);
        Assert.Equal(["Everyone"], Acl.Users["carol"]);
        Assert.Equal(["HR", "Finance", "Everyone", "Admin"], Acl.Users["admin"]);
    }

    [Fact]
    public void NonEveryoneRoles_StripsEveryone()
    {
        Assert.Equal(["HR"], Acl.NonEveryoneRoles(["HR", "Everyone"]));
        Assert.Empty(Acl.NonEveryoneRoles(["Everyone"]));
    }
}
