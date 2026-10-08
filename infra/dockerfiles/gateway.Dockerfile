# Gateway: ASP.NET Core over the published Release build.
#
# Build context is the REPO ROOT (see infra/docker-compose.yml). SDK
# image builds + publishes; the runtime image ships only the publish
# output (no SDK, no source). dotnet/global.json pins the SDK; the
# runtime tag tracks net10.0 (LTS line, not a specific patch — the
# gateway carries no numeric behavior that a runtime patch could skew).
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src
COPY dotnet/ ./
RUN dotnet restore TrustLayer.sln
RUN dotnet publish src/TrustLayer.Gateway/TrustLayer.Gateway.csproj \
    -c Release --no-restore -o /publish

FROM mcr.microsoft.com/dotnet/aspnet:10.0
WORKDIR /app
COPY --from=build /publish ./
EXPOSE 8080
ENV ASPNETCORE_URLS=http://+:8080
HEALTHCHECK --interval=10s --timeout=5s --retries=12 \
    CMD wget -qO- http://localhost:8080/health || exit 1
ENTRYPOINT ["dotnet", "TrustLayer.Gateway.dll"]
