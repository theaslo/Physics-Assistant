# Container Image Signing and Verification Service
FROM alpine:3.19

# Install cosign, crane, and other tools
RUN apk add --no-cache \
    bash \
    curl \
    git \
    jq \
    openssl \
    ca-certificates

# Install cosign for container signing
RUN curl -L https://github.com/sigstore/cosign/releases/latest/download/cosign-linux-amd64 -o /usr/local/bin/cosign && \
    chmod +x /usr/local/bin/cosign

# Install crane for container manipulation
RUN curl -L https://github.com/google/go-containerregistry/releases/latest/download/go-containerregistry_Linux_x86_64.tar.gz | \
    tar -xzC /usr/local/bin crane

# Install syft for SBOM generation
RUN curl -L https://github.com/anchore/syft/releases/latest/download/syft_linux_amd64.tar.gz | \
    tar -xzC /usr/local/bin syft

# Create directories
RUN mkdir -p /signing/{keys,policies,scripts,reports}

# Copy signing scripts and policies
COPY security/signing/scripts/ /signing/scripts/
COPY security/signing/policies/ /signing/policies/

# Set permissions
RUN chmod +x /signing/scripts/*.sh

# Create non-root user
RUN addgroup -g 1001 signer && \
    adduser -D -u 1001 -G signer signer && \
    chown -R signer:signer /signing

USER signer
WORKDIR /signing

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD cosign version || exit 1

CMD ["/signing/scripts/sign-images.sh"]