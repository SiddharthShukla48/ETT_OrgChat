"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useRouter } from "next/navigation"
import { useAuth } from "@/lib/auth-context"

export function LoginForm() {
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")
  const { login, isLoading } = useAuth()
  const router = useRouter()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!username || !password) {
      return
    }

    const success = await login({ username, password })
    
    if (success) {
      // Get user role from context after successful login
      const userData = JSON.parse(localStorage.getItem('user') || '{}')
      
      // Redirect based on role
      if (userData.role === "admin") {
        router.push("/admin")
      } else {
        router.push("/chat")
      }
    }
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="text-2xl text-center">Login</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="username">Username</Label>
            <Input
              id="username"
              type="text"
              placeholder="Enter your username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="password">Password</Label>
            <Input
              id="password"
              type="password"
              placeholder="Enter your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          <Button type="submit" className="w-full" disabled={isLoading}>
            {isLoading ? "Logging in..." : "Login"}
          </Button>
        </form>

        <div className="mt-6 text-sm text-muted-foreground">
          <h4 className="font-medium mb-2">Demo Credentials:</h4>
          <div className="space-y-1 text-xs">
            <div><strong>Admin:</strong> admin / admin123</div>
            <div><strong>HR:</strong> hr_manager / hr123</div>
            <div><strong>Manager:</strong> team_manager / manager123</div>
            <div><strong>Employee:</strong> employee1 / emp123</div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
