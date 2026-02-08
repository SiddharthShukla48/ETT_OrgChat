"use client"

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { apiClient, User, LoginCredentials } from '@/lib/api'
import { useToast } from '@/hooks/use-toast'

interface AuthContextType {
  user: User | null
  isLoading: boolean
  login: (credentials: LoginCredentials) => Promise<{ success: boolean; user?: User }>
  logout: () => void
  isAuthenticated: boolean
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

interface AuthProviderProps {
  children: ReactNode
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const { toast } = useToast()

  useEffect(() => {
    const initAuth = async () => {
      const token = localStorage.getItem('auth_token')
      const savedUser = localStorage.getItem('user')
      
      if (token && savedUser) {
        try {
          // Verify token is still valid by fetching current user
          const currentUser = await apiClient.getCurrentUser()
          setUser(currentUser)
          localStorage.setItem('user', JSON.stringify(currentUser))
        } catch (error) {
          // Token is invalid, clear it
          localStorage.removeItem('auth_token')
          localStorage.removeItem('user')
          apiClient.clearToken()
        }
      }
      setIsLoading(false)
    }

    initAuth()
  }, [])

  const login = async (credentials: LoginCredentials): Promise<{ success: boolean; user?: User }> => {
    try {
      setIsLoading(true)
      
      // Authenticate with backend
      await apiClient.login(credentials)
      
      // Get user details
      const userData = await apiClient.getCurrentUser()
      setUser(userData)
      
      // Store user data
      localStorage.setItem('user', JSON.stringify(userData))
      
      toast({
        title: "Login successful",
        description: `Welcome back, ${userData.username}!`,
      })
      
      return { success: true, user: userData }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Login failed'
      
      toast({
        title: "Login failed",
        description: errorMessage,
        variant: "destructive",
      })
      
      return { success: false }
    } finally {
      setIsLoading(false)
    }
  }

  const logout = () => {
    setUser(null)
    apiClient.clearToken()
    
    toast({
      title: "Logged out",
      description: "You have been successfully logged out.",
    })
  }

  const value: AuthContextType = {
    user,
    isLoading,
    login,
    logout,
    isAuthenticated: !!user,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
