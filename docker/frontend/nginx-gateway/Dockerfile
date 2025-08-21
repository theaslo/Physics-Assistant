# Nginx Gateway/Load Balancer for Physics Assistant
FROM nginx:alpine

LABEL maintainer="Physics Assistant Team"
LABEL description="Nginx gateway and load balancer for Physics Assistant platform"

# Install additional tools
RUN apk add --no-cache \
    curl \
    openssl \
    && rm -rf /var/cache/apk/*

# Create application user
RUN addgroup -g 1001 physicsapp && \
    adduser -D -u 1001 -G physicsapp physicsapp

# Copy nginx configuration
COPY docker/frontend/nginx-gateway/nginx.conf /etc/nginx/nginx.conf
COPY docker/frontend/nginx-gateway/default.conf /etc/nginx/conf.d/default.conf
COPY docker/frontend/nginx-gateway/ssl/ /etc/nginx/ssl/

# Create necessary directories
RUN mkdir -p /var/cache/nginx /var/log/nginx /usr/share/nginx/html && \
    chown -R physicsapp:physicsapp /var/cache/nginx /var/log/nginx /usr/share/nginx/html

# Copy static files and error pages
COPY docker/frontend/nginx-gateway/html/ /usr/share/nginx/html/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:80/health || exit 1

# Switch to application user
USER physicsapp

# Expose ports
EXPOSE 80 443

# Start nginx
CMD ["nginx", "-g", "daemon off;"]