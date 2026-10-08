using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;
using Microsoft.IdentityModel.Tokens;

namespace TrustLayer.Gateway.Auth;

public sealed class TokenService
{
    private readonly byte[] _key;
    private readonly string _issuer;

    public TokenService(string secret, string issuer = "trustlayer")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(secret);
        if (secret.Length < 32)
        {
            throw new ArgumentException("JWT secret must be at least 32 chars.", nameof(secret));
        }

        _key = Encoding.UTF8.GetBytes(secret);
        _issuer = issuer;
    }

    public string Issue(string username, string[] roles)
    {
        var claims = new List<Claim> { new(ClaimTypes.Name, username) };
        claims.AddRange(roles.Select(r => new Claim(ClaimTypes.Role, r)));
        var credentials = new SigningCredentials(
            new SymmetricSecurityKey(_key), SecurityAlgorithms.HmacSha256);
        var token = new JwtSecurityToken(
            issuer: _issuer,
            audience: _issuer,
            claims: claims,
            expires: DateTime.UtcNow.AddHours(8),
            signingCredentials: credentials);
        return new JwtSecurityTokenHandler().WriteToken(token);
    }

    public TokenValidationParameters ValidationParameters() => new()
    {
        ValidateIssuer = true,
        ValidateAudience = true,
        ValidateLifetime = true,
        ValidateIssuerSigningKey = true,
        ValidIssuer = _issuer,
        ValidAudience = _issuer,
        IssuerSigningKey = new SymmetricSecurityKey(_key),
        ClockSkew = TimeSpan.FromMinutes(1),
    };
}
