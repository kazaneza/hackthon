// src/utils/sessionManager.ts
import axios from 'axios';

class SessionManager {
  private static instance: SessionManager;
  private sessionId: string | null = null;
  private storageKey = 'bank_app_session_id';

  private constructor() {
    // Load session ID from localStorage on initialization
    this.loadSessionFromStorage();
  }

  public static getInstance(): SessionManager {
    if (!SessionManager.instance) {
      SessionManager.instance = new SessionManager();
    }
    return SessionManager.instance;
  }

  private loadSessionFromStorage(): void {
    try {
      const stored = localStorage.getItem(this.storageKey);
      if (stored) {
        console.log('Loaded session from storage:', stored);
        this.sessionId = stored;
      }
    } catch (error) {
      console.error('Error loading session from storage:', error);
    }
  }

  private saveSessionToStorage(): void {
    try {
      if (this.sessionId) {
        localStorage.setItem(this.storageKey, this.sessionId);
        console.log('Saved session to storage:', this.sessionId);
      } else {
        localStorage.removeItem(this.storageKey);
      }
    } catch (error) {
      console.error('Error saving session to storage:', error);
    }
  }

  public setSessionId(id: string): void {
    console.log('Setting session ID:', id);
    this.sessionId = id;
    this.saveSessionToStorage();
  }

  public getSessionId(): string | null {
    console.log('Getting session ID:', this.sessionId);
    return this.sessionId;
  }

  public clearSession(): void {
    console.log('Clearing session');
    this.sessionId = null;
    this.saveSessionToStorage();
  }

  public createApiInstance(baseURL: string): axios.AxiosInstance {
    const instance = axios.create({ baseURL });

    // Request interceptor - add session ID to every request
    instance.interceptors.request.use((config) => {
      if (this.sessionId) {
        // For NLP API, use user_id; for main API, use X-Session-ID
        if (baseURL.includes('8888')) {
          // This is the NLP API - add user_id to data if it's a POST
          if (config.method === 'post' && config.data) {
            config.data.user_id = this.sessionId;
          }
        } else {
          // This is the main API - use header
          config.headers['X-Session-ID'] = this.sessionId;
        }
        console.log('Adding session ID to request:', this.sessionId);
      }
      return config;
    });

    // Response interceptor - capture session ID from responses
    instance.interceptors.response.use((response) => {
      const newSessionId = response.headers['x-session-id'] ?? response.headers['X-Session-ID'];
      if (newSessionId && newSessionId !== this.sessionId) {
        console.log('Received new session ID:', newSessionId);
        this.setSessionId(newSessionId);
      }
      return response;
    });

    return instance;
  }
}

export const sessionManager = SessionManager.getInstance();

// Create API instances with session management
export const api = sessionManager.createApiInstance('http://localhost:8000');
export const nlpApi = sessionManager.createApiInstance('http://localhost:8888');