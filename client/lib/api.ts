// API configuration for connecting to FastAPI backend
const API_BASE_URL = typeof window !== 'undefined' 
  ? (window as any).NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  : 'http://localhost:8000'

export interface LoginCredentials {
  username: string
  password: string
}

export interface User {
  id: string
  username: string
  role: 'employee' | 'manager' | 'hr' | 'admin'
  is_active: boolean
  created_at: string
  updated_at: string
}

export interface AuthResponse {
  access_token: string
  token_type: string
}

export interface CreateUserData {
  username: string
  password: string
  role: 'employee' | 'manager' | 'hr' | 'admin'
  is_active?: boolean
}

class ApiClient {
  private baseUrl: string
  private token: string | null = null

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl
    
    // Load token from localStorage if available
    if (typeof window !== 'undefined') {
      this.token = localStorage.getItem('auth_token')
    }
  }

  setToken(token: string) {
    this.token = token
    if (typeof window !== 'undefined') {
      localStorage.setItem('auth_token', token)
    }
  }

  clearToken() {
    this.token = null
    if (typeof window !== 'undefined') {
      localStorage.removeItem('auth_token')
      localStorage.removeItem('user')
    }
  }

  private async makeRequest<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`
    
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(options.headers as Record<string, string> || {}),
    }

    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`
    }

    const response = await fetch(url, {
      ...options,
      headers,
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
    }

    return response.json()
  }

  // Authentication endpoints
  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    const formData = new FormData()
    formData.append('username', credentials.username)
    formData.append('password', credentials.password)

    const response = await fetch(`${this.baseUrl}/auth/login`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || 'Login failed')
    }

    const authData = await response.json()
    this.setToken(authData.access_token)
    return authData
  }

  // User endpoints
  async getCurrentUser(): Promise<User> {
    return this.makeRequest<User>('/users/me')
  }

  async getUsers(skip: number = 0, limit: number = 100): Promise<User[]> {
    return this.makeRequest<User[]>(`/users/?skip=${skip}&limit=${limit}`)
  }

  async createUser(userData: CreateUserData): Promise<User> {
    return this.makeRequest<User>('/users/', {
      method: 'POST',
      body: JSON.stringify(userData),
    })
  }

  async updateUser(userId: string, userData: Partial<CreateUserData>): Promise<User> {
    return this.makeRequest<User>(`/users/${userId}`, {
      method: 'PUT',
      body: JSON.stringify(userData),
    })
  }

  async deleteUser(userId: string): Promise<{ message: string }> {
    return this.makeRequest<{ message: string }>(`/users/${userId}`, {
      method: 'DELETE',
    })
  }

  async deactivateUser(userId: string): Promise<{ message: string }> {
    return this.makeRequest<{ message: string }>(`/users/${userId}/deactivate`, {
      method: 'PATCH',
    })
  }

  async activateUser(userId: string): Promise<{ message: string }> {
    return this.makeRequest<{ message: string }>(`/users/${userId}/activate`, {
      method: 'PATCH',
    })
  }

  // Health check
  async healthCheck(): Promise<{ status: string }> {
    return this.makeRequest<{ status: string }>('/health')
  }

  // Multi-Agent RAG Chat endpoints
  async multiAgentChat(message: string, sessionId?: string): Promise<{
    response: string
    session_id: string
    agent_used?: string
    query_analysis?: any
  }> {
    return this.makeRequest('/api/chat/multi-agent', {
      method: 'POST',
      body: JSON.stringify({
        message,
        session_id: sessionId
      })
    })
  }

  async getChatHistory(sessionId: string): Promise<{
    session_id: string
    history: any[]
  }> {
    return this.makeRequest(`/api/chat/sessions/${sessionId}/history`)
  }

  async clearChatSession(sessionId: string): Promise<{ message: string }> {
    return this.makeRequest(`/api/chat/sessions/${sessionId}`, {
      method: 'DELETE'
    })
  }

  async chatHealthCheck(): Promise<{
    status: string
    multi_agent_rag: string
    test_response_length: number
  }> {
    return this.makeRequest('/api/chat/health')
  }
}

export const apiClient = new ApiClient()
export default ApiClient
