"use client"

import React, { useState, useEffect } from 'react'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useToast } from "@/hooks/use-toast"
import { Badge } from "@/components/ui/badge"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Plus, Trash2, Edit, Users, UserX, UserCheck } from "lucide-react"
import { apiClient, User, CreateUserData } from '@/lib/api'

type UserRole = 'employee' | 'manager' | 'hr' | 'admin'

interface CreateUserForm {
  username: string
  password: string
  role: UserRole
  is_active: boolean
}

export function UserManagement() {
  const [users, setUsers] = useState<User[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false)
  const [createUserForm, setCreateUserForm] = useState<CreateUserForm>({
    username: '',
    password: '',
    role: 'employee',
    is_active: true
  })
  const { toast } = useToast()

  // Fetch users from backend
  const fetchUsers = async () => {
    try {
      setIsLoading(true)
      const usersData = await apiClient.getUsers()
      setUsers(usersData)
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to fetch users",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    fetchUsers()
  }, [])

  // Create new user
  const handleCreateUser = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!createUserForm.username || !createUserForm.password) {
      toast({
        title: "Validation Error",
        description: "Username and password are required",
        variant: "destructive",
      })
      return
    }

    try {
      const userData: CreateUserData = {
        username: createUserForm.username,
        password: createUserForm.password,
        role: createUserForm.role,
        is_active: createUserForm.is_active
      }

      await apiClient.createUser(userData)
      
      toast({
        title: "Success",
        description: `User ${createUserForm.username} created successfully`,
      })

      // Reset form and close dialog
      setCreateUserForm({
        username: '',
        password: '',
        role: 'employee',
        is_active: true
      })
      setIsCreateDialogOpen(false)
      
      // Refresh users list
      fetchUsers()
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to create user",
        variant: "destructive",
      })
    }
  }

  // Permanently delete user
  const handleDeleteUser = async (userId: string, username: string) => {
    if (!confirm(`Are you sure you want to PERMANENTLY DELETE user "${username}"? This action cannot be undone!`)) {
      return
    }

    try {
      await apiClient.deleteUser(userId)
      
      toast({
        title: "Success",
        description: `User ${username} permanently deleted`,
      })
      
      // Refresh users list
      fetchUsers()
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to delete user",
        variant: "destructive",
      })
    }
  }

  // Deactivate user
  const handleDeactivateUser = async (userId: string, username: string) => {
    if (!confirm(`Are you sure you want to deactivate user "${username}"?`)) {
      return
    }

    try {
      await apiClient.deactivateUser(userId)
      
      toast({
        title: "Success",
        description: `User ${username} deactivated successfully`,
      })
      
      // Refresh users list
      fetchUsers()
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to deactivate user",
        variant: "destructive",
      })
    }
  }

  // Activate user
  const handleActivateUser = async (userId: string, username: string) => {
    if (!confirm(`Are you sure you want to activate user "${username}"?`)) {
      return
    }

    try {
      await apiClient.activateUser(userId)
      
      toast({
        title: "Success",
        description: `User ${username} activated successfully`,
      })
      
      // Refresh users list
      fetchUsers()
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to activate user",
        variant: "destructive",
      })
    }
  }

  // Get role badge color
  const getRoleBadgeVariant = (role: string) => {
    switch (role) {
      case 'admin': return 'destructive'
      case 'hr': return 'default'
      case 'manager': return 'secondary'
      case 'employee': return 'outline'
      default: return 'outline'
    }
  }

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            User Management
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">Loading users...</div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Users className="h-5 w-5" />
          User Management
        </CardTitle>
        <div className="flex justify-between items-center">
          <p className="text-sm text-muted-foreground">
            Manage users, roles, and permissions
          </p>
          <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
            <DialogTrigger asChild>
              <Button>
                <Plus className="h-4 w-4 mr-2" />
                Add User
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create New User</DialogTitle>
                <DialogDescription>
                  Add a new user to the system with appropriate role and permissions.
                </DialogDescription>
              </DialogHeader>
              <form onSubmit={handleCreateUser} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="username">Username</Label>
                  <Input
                    id="username"
                    value={createUserForm.username}
                    onChange={(e) => setCreateUserForm(prev => ({ ...prev, username: e.target.value }))}
                    placeholder="Enter username"
                    required
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="password">Password</Label>
                  <Input
                    id="password"
                    type="password"
                    value={createUserForm.password}
                    onChange={(e) => setCreateUserForm(prev => ({ ...prev, password: e.target.value }))}
                    placeholder="Enter password"
                    required
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="role">Role</Label>
                  <Select 
                    value={createUserForm.role} 
                    onValueChange={(value: UserRole) => setCreateUserForm(prev => ({ ...prev, role: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select role" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="employee">Employee</SelectItem>
                      <SelectItem value="manager">Manager</SelectItem>
                      <SelectItem value="hr">HR</SelectItem>
                      <SelectItem value="admin">Admin</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="flex justify-end gap-2">
                  <Button type="button" variant="outline" onClick={() => setIsCreateDialogOpen(false)}>
                    Cancel
                  </Button>
                  <Button type="submit">Create User</Button>
                </div>
              </form>
            </DialogContent>
          </Dialog>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="text-sm text-muted-foreground">
            Total Users: {users.length}
          </div>
          
          {users.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              No users found
            </div>
          ) : (
            <div className="space-y-2">
              {users.map((user) => (
                <div key={user.id} className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="flex-1">
                    <div className="flex items-center gap-3">
                      <div>
                        <h4 className="font-medium">{user.username}</h4>
                        <p className="text-sm text-muted-foreground">
                          Created: {new Date(user.created_at).toLocaleDateString()}
                        </p>
                      </div>
                      <Badge variant={getRoleBadgeVariant(user.role)}>
                        {user.role.toUpperCase()}
                      </Badge>
                      <Badge variant={user.is_active ? "default" : "destructive"}>
                        {user.is_active ? "Active" : "Inactive"}
                      </Badge>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    {user.is_active ? (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleDeactivateUser(user.id, user.username)}
                        className="text-yellow-600 hover:text-yellow-700"
                        title="Deactivate User (Reversible)"
                      >
                        <UserX className="h-4 w-4" />
                      </Button>
                    ) : (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleActivateUser(user.id, user.username)}
                        className="text-green-600 hover:text-green-700"
                        title="Activate User"
                      >
                        <UserCheck className="h-4 w-4" />
                      </Button>
                    )}
                    
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleDeleteUser(user.id, user.username)}
                      className="text-destructive hover:text-destructive"
                      title="Delete User Permanently"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
