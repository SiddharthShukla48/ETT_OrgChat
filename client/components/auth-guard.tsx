"use client"

import type React from "react"

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import { useAuth } from "@/lib/auth-context"

interface AuthGuardProps {
  children: React.ReactNode
  allowedRoles?: ("employee" | "manager" | "hr" | "admin")[]
}

export function AuthGuard({ children, allowedRoles }: AuthGuardProps) {
  const { user, isLoading } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!isLoading && !user) {
      router.push("/")
      return
    }

    if (user && allowedRoles && !allowedRoles.includes(user.role)) {
      router.push("/")
      return
    }
  }, [user, isLoading, router, allowedRoles])

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    )
  }

  if (!user) {
    return null
  }

  return <>{children}</>
}

// Export useAuth for backward compatibility
export { useAuth } from "@/lib/auth-context"
