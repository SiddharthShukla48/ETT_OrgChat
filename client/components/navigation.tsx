"use client"

import { useAuth } from "@/components/auth-guard"
import { Button } from "@/components/ui/button"
import { useRouter, usePathname } from "next/navigation"
import { MessageSquare, Settings, LogOut, Bot, Users } from "lucide-react"

export function Navigation() {
  const { user, logout } = useAuth()
  const router = useRouter()
  const pathname = usePathname()

  if (!user) return null

  const navigationItems = [
    {
      label: "Chat",
      icon: MessageSquare,
      href: "/chat",
      roles: ["employee", "manager", "admin"],
    },
    {
      label: "Admin Panel",
      icon: Settings,
      href: "/admin",
      roles: ["admin"],
    },
  ]

  const availableItems = navigationItems.filter((item) => item.roles.includes(user.role))

  return (
    <nav className="bg-card border-b border-border p-4">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-3">
            <Bot className="h-8 w-8 text-primary" />
            <div>
              <h1 className="text-xl font-semibold text-card-foreground">OrgChat</h1>
              {/* <p className="text-sm text-muted-foreground">
                {user.email} ({user.role})
              </p> */}
            </div>
          </div>

          {/* Navigation Links */}
          <div className="flex items-center gap-2">
            {availableItems.map((item) => {
              const Icon = item.icon
              const isActive = pathname === item.href
              return (
                <Button
                  key={item.href}
                  variant={isActive ? "default" : "ghost"}
                  onClick={() => router.push(item.href)}
                  className="flex items-center gap-2"
                >
                  <Icon className="h-4 w-4" />
                  {item.label}
                </Button>
              )
            })}
          </div>
        </div>

        {/* User Actions */}
        <div className="flex items-center gap-2">
          {user.role === "admin" && (
            <div className="flex items-center gap-1 text-xs text-muted-foreground mr-4">
              <Users className="h-3 w-3" />
              <span>Admin Access</span>
            </div>
          )}
          <Button variant="outline" onClick={logout} className="flex items-center gap-2 bg-transparent">
            <LogOut className="h-4 w-4" />
            Logout
          </Button>
        </div>
      </div>
    </nav>
  )
}
