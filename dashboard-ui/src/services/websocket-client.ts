/**
 * WebSocket Client for Real-time Dashboard Updates
 * Handles WebSocket connections and real-time data streaming
 */

import { 
  WebSocketMessage, 
  WebSocketMessageType, 
  MetricsUpdate, 
  Alert, 
  StudentProgressUpdate 
} from '../types/api';

export interface WebSocketConfig {
  url: string;
  reconnectAttempts?: number;
  reconnectInterval?: number;
  heartbeatInterval?: number;
  autoReconnect?: boolean;
}

export interface WebSocketCallbacks {
  onOpen?: () => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (error: Event) => void;
  onMessage?: (message: WebSocketMessage) => void;
  onMetricsUpdate?: (data: MetricsUpdate) => void;
  onAlert?: (data: Alert) => void;
  onStudentProgress?: (data: StudentProgressUpdate) => void;
  onHeartbeat?: () => void;
  onReconnect?: (attempt: number) => void;
  onReconnectFailed?: () => void;
}

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private config: Required<WebSocketConfig>;
  private callbacks: WebSocketCallbacks;
  private reconnectTimer: number | null = null;
  private heartbeatTimer: number | null = null;
  private currentReconnectAttempt: number = 0;
  private isConnected: boolean = false;
  private isDestroyed: boolean = false;
  private lastHeartbeat: number = 0;

  constructor(config: WebSocketConfig, callbacks: WebSocketCallbacks = {}) {
    this.config = {
      url: config.url,
      reconnectAttempts: config.reconnectAttempts ?? 5,
      reconnectInterval: config.reconnectInterval ?? 3000,
      heartbeatInterval: config.heartbeatInterval ?? 30000,
      autoReconnect: config.autoReconnect ?? true,
    };
    this.callbacks = callbacks;
  }

  // ============================================================================
  // Connection Management
  // ============================================================================

  public connect(): void {
    if (this.isDestroyed) {
      console.warn('WebSocket client is destroyed, cannot connect');
      return;
    }

    if (this.ws?.readyState === WebSocket.CONNECTING || this.isConnected) {
      console.warn('WebSocket already connecting or connected');
      return;
    }

    try {
      console.log(`Connecting to WebSocket: ${this.config.url}`);
      this.ws = new WebSocket(this.config.url);
      this.setupEventListeners();
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      this.callbacks.onError?.(error as Event);
      this.handleReconnect();
    }
  }

  public disconnect(): void {
    this.isDestroyed = true;
    this.clearTimers();
    
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    
    this.isConnected = false;
    console.log('WebSocket disconnected');
  }

  public send(message: any): boolean {
    if (!this.isConnected || !this.ws) {
      console.warn('WebSocket not connected, cannot send message:', message);
      return false;
    }

    try {
      const messageString = typeof message === 'string' ? message : JSON.stringify(message);
      this.ws.send(messageString);
      return true;
    } catch (error) {
      console.error('Failed to send WebSocket message:', error);
      return false;
    }
  }

  public getConnectionState(): string {
    if (!this.ws) return 'CLOSED';
    
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING: return 'CONNECTING';
      case WebSocket.OPEN: return 'OPEN';
      case WebSocket.CLOSING: return 'CLOSING';
      case WebSocket.CLOSED: return 'CLOSED';
      default: return 'UNKNOWN';
    }
  }

  public isConnectionOpen(): boolean {
    return this.isConnected && this.ws?.readyState === WebSocket.OPEN;
  }

  // ============================================================================
  // Event Listeners Setup
  // ============================================================================

  private setupEventListeners(): void {
    if (!this.ws) return;

    this.ws.onopen = (event) => {
      console.log('WebSocket connected successfully');
      this.isConnected = true;
      this.currentReconnectAttempt = 0;
      this.startHeartbeat();
      this.callbacks.onOpen?.();
    };

    this.ws.onclose = (event) => {
      console.log('WebSocket connection closed:', event.code, event.reason);
      this.isConnected = false;
      this.clearTimers();
      this.callbacks.onClose?.(event);
      
      if (!this.isDestroyed && this.config.autoReconnect && event.code !== 1000) {
        this.handleReconnect();
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.callbacks.onError?.(error);
    };

    this.ws.onmessage = (event) => {
      this.handleMessage(event.data);
    };
  }

  // ============================================================================
  // Message Handling
  // ============================================================================

  private handleMessage(data: string): void {
    try {
      const message: WebSocketMessage = JSON.parse(data);
      
      // Update last heartbeat time
      if (message.type === 'heartbeat') {
        this.lastHeartbeat = Date.now();
        this.callbacks.onHeartbeat?.();
        return;
      }

      // Call general message callback
      this.callbacks.onMessage?.(message);

      // Call specific callbacks based on message type
      switch (message.type) {
        case 'metrics_update':
          this.callbacks.onMetricsUpdate?.(message.data as MetricsUpdate);
          break;

        case 'alert':
          this.callbacks.onAlert?.(message.data as Alert);
          break;

        case 'student_progress':
          this.callbacks.onStudentProgress?.(message.data as StudentProgressUpdate);
          break;

        case 'connection_status':
          console.log('Connection status update:', message.data);
          break;

        default:
          console.warn('Unknown WebSocket message type:', message.type);
      }
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error, data);
    }
  }

  // ============================================================================
  // Reconnection Logic
  // ============================================================================

  private handleReconnect(): void {
    if (this.isDestroyed || !this.config.autoReconnect) {
      return;
    }

    if (this.currentReconnectAttempt >= this.config.reconnectAttempts) {
      console.error('Max reconnection attempts reached');
      this.callbacks.onReconnectFailed?.();
      return;
    }

    this.currentReconnectAttempt++;
    const delay = this.config.reconnectInterval * Math.pow(1.5, this.currentReconnectAttempt - 1);
    
    console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.currentReconnectAttempt}/${this.config.reconnectAttempts})`);
    
    this.reconnectTimer = window.setTimeout(() => {
      this.callbacks.onReconnect?.(this.currentReconnectAttempt);
      this.connect();
    }, delay);
  }

  // ============================================================================
  // Heartbeat Management
  // ============================================================================

  private startHeartbeat(): void {
    this.lastHeartbeat = Date.now();
    
    this.heartbeatTimer = window.setInterval(() => {
      if (!this.isConnected) {
        this.clearTimers();
        return;
      }

      // Check if we've missed heartbeats
      const timeSinceLastHeartbeat = Date.now() - this.lastHeartbeat;
      if (timeSinceLastHeartbeat > this.config.heartbeatInterval * 2) {
        console.warn('Heartbeat timeout, reconnecting...');
        this.ws?.close(1006, 'Heartbeat timeout');
        return;
      }

      // Send ping if supported
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.send({ type: 'ping', timestamp: Date.now() });
      }
    }, this.config.heartbeatInterval);
  }

  private clearTimers(): void {
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    
    if (this.heartbeatTimer !== null) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  // ============================================================================
  // Utility Methods
  // ============================================================================

  public getStats(): {
    isConnected: boolean;
    connectionState: string;
    reconnectAttempts: number;
    lastHeartbeat: number;
    timeSinceLastHeartbeat: number;
  } {
    return {
      isConnected: this.isConnected,
      connectionState: this.getConnectionState(),
      reconnectAttempts: this.currentReconnectAttempt,
      lastHeartbeat: this.lastHeartbeat,
      timeSinceLastHeartbeat: Date.now() - this.lastHeartbeat,
    };
  }

  public updateCallbacks(newCallbacks: Partial<WebSocketCallbacks>): void {
    this.callbacks = { ...this.callbacks, ...newCallbacks };
  }

  public updateConfig(newConfig: Partial<WebSocketConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }
}

// ============================================================================
// WebSocket Hook for React
// ============================================================================

import { useEffect, useRef, useState } from 'react';

export interface UseWebSocketOptions extends WebSocketConfig {
  enabled?: boolean;
  callbacks?: WebSocketCallbacks;
}

export interface UseWebSocketReturn {
  client: WebSocketClient | null;
  isConnected: boolean;
  connectionState: string;
  send: (message: any) => boolean;
  disconnect: () => void;
  connect: () => void;
  stats: ReturnType<WebSocketClient['getStats']> | null;
}

export function useWebSocket(options: UseWebSocketOptions): UseWebSocketReturn {
  const clientRef = useRef<WebSocketClient | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionState, setConnectionState] = useState('CLOSED');
  const [stats, setStats] = useState<ReturnType<WebSocketClient['getStats']> | null>(null);

  const enabled = options.enabled ?? true;

  useEffect(() => {
    if (!enabled || !options.url) {
      return;
    }

    const callbacks: WebSocketCallbacks = {
      ...options.callbacks,
      onOpen: () => {
        setIsConnected(true);
        setConnectionState('OPEN');
        options.callbacks?.onOpen?.();
      },
      onClose: (event) => {
        setIsConnected(false);
        setConnectionState('CLOSED');
        options.callbacks?.onClose?.(event);
      },
      onError: (error) => {
        setConnectionState('ERROR');
        options.callbacks?.onError?.(error);
      },
    };

    clientRef.current = new WebSocketClient(options, callbacks);
    clientRef.current.connect();

    // Update stats periodically
    const statsInterval = setInterval(() => {
      if (clientRef.current) {
        setStats(clientRef.current.getStats());
        setConnectionState(clientRef.current.getConnectionState());
      }
    }, 1000);

    return () => {
      clearInterval(statsInterval);
      if (clientRef.current) {
        clientRef.current.disconnect();
        clientRef.current = null;
      }
    };
  }, [enabled, options.url]);

  const send = (message: any) => {
    return clientRef.current?.send(message) ?? false;
  };

  const disconnect = () => {
    clientRef.current?.disconnect();
  };

  const connect = () => {
    clientRef.current?.connect();
  };

  return {
    client: clientRef.current,
    isConnected,
    connectionState,
    send,
    disconnect,
    connect,
    stats,
  };
}

// ============================================================================
// Server-Sent Events (SSE) Client
// ============================================================================

export interface SSEConfig {
  url: string;
  withCredentials?: boolean;
  retryInterval?: number;
  maxRetries?: number;
}

export interface SSECallbacks {
  onOpen?: () => void;
  onMessage?: (data: any, event: MessageEvent) => void;
  onError?: (error: Event) => void;
  onDashboardUpdate?: (data: any) => void;
  onHeartbeat?: () => void;
}

export class SSEClient {
  private eventSource: EventSource | null = null;
  private config: SSEConfig;
  private callbacks: SSECallbacks;
  private retryCount: number = 0;
  private retryTimer: number | null = null;

  constructor(config: SSEConfig, callbacks: SSECallbacks = {}) {
    this.config = {
      retryInterval: 3000,
      maxRetries: 5,
      ...config,
    };
    this.callbacks = callbacks;
  }

  public connect(): void {
    try {
      this.eventSource = new EventSource(this.config.url, {
        withCredentials: this.config.withCredentials,
      });

      this.eventSource.onopen = () => {
        console.log('SSE connected');
        this.retryCount = 0;
        this.callbacks.onOpen?.();
      };

      this.eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.callbacks.onMessage?.(data, event);
        } catch (error) {
          console.error('Failed to parse SSE message:', error);
        }
      };

      this.eventSource.onerror = (error) => {
        console.error('SSE error:', error);
        this.callbacks.onError?.(error);
        this.handleReconnect();
      };

      // Custom event listeners
      this.eventSource.addEventListener('dashboard_update', (event) => {
        try {
          const data = JSON.parse(event.data);
          this.callbacks.onDashboardUpdate?.(data);
        } catch (error) {
          console.error('Failed to parse dashboard update:', error);
        }
      });

      this.eventSource.addEventListener('heartbeat', () => {
        this.callbacks.onHeartbeat?.();
      });

    } catch (error) {
      console.error('Failed to create SSE connection:', error);
      this.handleReconnect();
    }
  }

  public disconnect(): void {
    if (this.retryTimer) {
      clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
    
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
  }

  private handleReconnect(): void {
    if (this.retryCount >= (this.config.maxRetries ?? 5)) {
      console.error('Max SSE retry attempts reached');
      return;
    }

    this.retryCount++;
    const delay = (this.config.retryInterval ?? 3000) * this.retryCount;

    console.log(`Retrying SSE connection in ${delay}ms (attempt ${this.retryCount})`);
    
    this.retryTimer = window.setTimeout(() => {
      this.connect();
    }, delay);
  }

  public getReadyState(): number | null {
    return this.eventSource?.readyState ?? null;
  }
}

export default WebSocketClient;