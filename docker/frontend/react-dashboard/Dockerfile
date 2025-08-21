# Multi-stage build for React Dashboard
FROM node:20-alpine AS builder

LABEL maintainer="Physics Assistant Team"
LABEL description="React analytics dashboard for Physics Assistant platform"

# Set working directory
WORKDIR /app

# Copy package files
COPY dashboard-ui/package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY dashboard-ui/ ./

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Install curl for health checks
RUN apk add --no-cache curl

# Create application user
RUN addgroup -g 1001 physicsapp && \
    adduser -D -u 1001 -G physicsapp physicsapp

# Copy built application from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom nginx configuration
COPY docker/frontend/react-dashboard/nginx.conf /etc/nginx/nginx.conf
COPY docker/frontend/react-dashboard/default.conf /etc/nginx/conf.d/default.conf

# Create necessary directories
RUN mkdir -p /var/cache/nginx /var/log/nginx && \
    chown -R physicsapp:physicsapp /var/cache/nginx /var/log/nginx && \
    chown -R physicsapp:physicsapp /usr/share/nginx/html

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5173/ || exit 1

# Switch to application user
USER physicsapp

# Expose port
EXPOSE 5173

# Start nginx
CMD ["nginx", "-g", "daemon off;"]