"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import { useAuth } from "@/components/auth-guard"
import { useToast } from "@/hooks/use-toast"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Navigation } from "@/components/navigation"
import { UserManagement } from "@/components/user-management"
import {
  Users,
  Database,
  Plus,
  Trash2,
  Briefcase,
  Calendar,
  DollarSign,
  TrendingUp,
  Building,
  FileText,
} from "lucide-react"

interface AdminUser {
  id: string
  email: string
  role: "employee" | "manager" | "admin"
  createdAt: Date
  status: "active" | "inactive"
}

interface PersonalInfo {
  id: string
  employeeId: string
  name: string
  email: string
  phone: string
  address: string
  emergencyContact: string
  emergencyPhone: string
  dateOfBirth: string
  gender: string
  nationality: string
  dateOfJoining: string
}

interface JobInfo {
  id: string
  employeeId: string
  designation: string
  department: string
  reportingManager: string
  location: string
  workSchedule: string
  shiftTimings: string
}

interface AttendanceLeave {
  id: string
  employeeId: string
  casualLeave: number
  sickLeave: number
  earnedLeave: number
  leaveHistory: string
  attendanceRecord: string
  holidayCalendar: string
}

interface PayrollInfo {
  id: string
  employeeId: string
  basicSalary: number
  hra: number
  allowances: number
  deductions: number
  bonuses: number
  reimbursements: number
  taxDeductions: number
  pf: number
  esi: number
}

interface PerformanceInfo {
  id: string
  employeeId: string
  performanceRating: string
  goals: string
  achievements: string
  trainingRecords: string
  promotions: string
}

interface AdminAssetInfo {
  id: string
  employeeId: string
  idCardInfo: string
  assignedLaptop: string
  seatingLocation: string
  buildingAccess: string
  travelHistory: string
}

interface HRPolicyInfo {
  id: string
  employeeId: string
  companyPolicies: string
  codeOfConduct: string
  complianceForms: string
  mandatoryCertifications: string
}

export function AdminPanel() {
  const { user } = useAuth()
  const { toast } = useToast()

  // User Management State
  const [users, setUsers] = useState<AdminUser[]>([
    {
      id: "1",
      email: "john.doe@company.com",
      role: "employee",
      createdAt: new Date("2024-01-15"),
      status: "active",
    },
    {
      id: "2",
      email: "jane.smith@company.com",
      role: "manager",
      createdAt: new Date("2024-02-01"),
      status: "active",
    },
    {
      id: "3",
      email: "admin@company.com",
      role: "admin",
      createdAt: new Date("2024-01-01"),
      status: "active",
    },
  ])

  const [newUser, setNewUser] = useState({
    email: "",
    role: "employee" as "employee" | "manager" | "admin",
  })

  // Employee Data States
  const [personalInfoData, setPersonalInfoData] = useState<PersonalInfo[]>([
    {
      id: "1",
      employeeId: "EMP001",
      name: "John Doe",
      email: "john.doe@company.com",
      phone: "+1-555-0123",
      address: "123 Main St, City, State 12345",
      emergencyContact: "Jane Doe",
      emergencyPhone: "+1-555-0124",
      dateOfBirth: "1990-05-15",
      gender: "Male",
      nationality: "American",
      dateOfJoining: "2024-01-15",
    },
  ])

  const [jobInfoData, setJobInfoData] = useState<JobInfo[]>([
    {
      id: "1",
      employeeId: "EMP001",
      designation: "Software Engineer",
      department: "Engineering",
      reportingManager: "Jane Smith",
      location: "New York Office",
      workSchedule: "Full-time",
      shiftTimings: "9:00 AM - 6:00 PM",
    },
  ])

  const [attendanceLeaveData, setAttendanceLeaveData] = useState<AttendanceLeave[]>([
    {
      id: "1",
      employeeId: "EMP001",
      casualLeave: 12,
      sickLeave: 10,
      earnedLeave: 15,
      leaveHistory: "2 days sick leave in March 2024",
      attendanceRecord: "98% attendance rate",
      holidayCalendar: "Standard company holidays",
    },
  ])

  const [payrollData, setPayrollData] = useState<PayrollInfo[]>([
    {
      id: "1",
      employeeId: "EMP001",
      basicSalary: 75000,
      hra: 15000,
      allowances: 5000,
      deductions: 2000,
      bonuses: 10000,
      reimbursements: 1500,
      taxDeductions: 18000,
      pf: 9000,
      esi: 1200,
    },
  ])

  const [performanceData, setPerformanceData] = useState<PerformanceInfo[]>([
    {
      id: "1",
      employeeId: "EMP001",
      performanceRating: "Exceeds Expectations",
      goals: "Complete 3 major projects, improve code quality",
      achievements: "Led successful product launch",
      trainingRecords: "React Advanced Course, Leadership Training",
      promotions: "Promoted to Senior Engineer in 2023",
    },
  ])

  const [adminAssetData, setAdminAssetData] = useState<AdminAssetInfo[]>([
    {
      id: "1",
      employeeId: "EMP001",
      idCardInfo: "ID: EMP001, Access Level: Standard",
      assignedLaptop: "MacBook Pro 16-inch, Serial: ABC123",
      seatingLocation: "Floor 3, Desk 45",
      buildingAccess: "Main building, Conference rooms",
      travelHistory: "Business trip to SF - March 2024",
    },
  ])

  const [hrPolicyData, setHRPolicyData] = useState<HRPolicyInfo[]>([
    {
      id: "1",
      employeeId: "EMP001",
      companyPolicies: "Remote work policy acknowledged",
      codeOfConduct: "Signed and acknowledged",
      complianceForms: "All forms completed",
      mandatoryCertifications: "Security training completed",
    },
  ])

  // Form states for each data type
  const [newPersonalInfo, setNewPersonalInfo] = useState<Partial<PersonalInfo>>({})
  const [newJobInfo, setNewJobInfo] = useState<Partial<JobInfo>>({})
  const [newAttendanceLeave, setNewAttendanceLeave] = useState<Partial<AttendanceLeave>>({})
  const [newPayroll, setNewPayroll] = useState<Partial<PayrollInfo>>({})
  const [newPerformance, setNewPerformance] = useState<Partial<PerformanceInfo>>({})
  const [newAdminAsset, setNewAdminAsset] = useState<Partial<AdminAssetInfo>>({})
  const [newHRPolicy, setNewHRPolicy] = useState<Partial<HRPolicyInfo>>({})

  // User Management Functions
  const handleCreateUser = (e: React.FormEvent) => {
    e.preventDefault()
    if (!newUser.email) {
      toast({
        title: "Error",
        description: "Email is required",
        variant: "destructive",
      })
      return
    }

    const user: AdminUser = {
      id: Date.now().toString(),
      email: newUser.email,
      role: newUser.role,
      createdAt: new Date(),
      status: "active",
    }

    setUsers((prev) => [...prev, user])
    setNewUser({ email: "", role: "employee" })

    toast({
      title: "User Created",
      description: `Successfully created user: ${user.email}`,
    })
  }

  const handleDeleteUser = (userId: string) => {
    const userToDelete = users.find((u) => u.id === userId)
    if (userToDelete?.role === "admin" && users.filter((u) => u.role === "admin").length === 1) {
      toast({
        title: "Cannot Delete",
        description: "Cannot delete the last admin user",
        variant: "destructive",
      })
      return
    }

    setUsers((prev) => prev.filter((u) => u.id !== userId))
    toast({
      title: "User Deleted",
      description: "User has been successfully deleted",
    })
  }

  // CRUD functions for each data type
  const handleCreatePersonalInfo = (e: React.FormEvent) => {
    e.preventDefault()
    if (!newPersonalInfo.employeeId || !newPersonalInfo.name || !newPersonalInfo.email) {
      toast({ title: "Error", description: "Employee ID, Name, and Email are required", variant: "destructive" })
      return
    }

    const data: PersonalInfo = {
      id: Date.now().toString(),
      employeeId: newPersonalInfo.employeeId || "",
      name: newPersonalInfo.name || "",
      email: newPersonalInfo.email || "",
      phone: newPersonalInfo.phone || "",
      address: newPersonalInfo.address || "",
      emergencyContact: newPersonalInfo.emergencyContact || "",
      emergencyPhone: newPersonalInfo.emergencyPhone || "",
      dateOfBirth: newPersonalInfo.dateOfBirth || "",
      gender: newPersonalInfo.gender || "",
      nationality: newPersonalInfo.nationality || "",
      dateOfJoining: newPersonalInfo.dateOfJoining || "",
    }

    setPersonalInfoData((prev) => [...prev, data])
    setNewPersonalInfo({})
    toast({ title: "Success", description: "Personal information added successfully" })
  }

  const handleDeletePersonalInfo = (id: string) => {
    setPersonalInfoData((prev) => prev.filter((item) => item.id !== id))
    toast({ title: "Success", description: "Personal information deleted" })
  }

  const handleCreateJobInfo = (e: React.FormEvent) => {
    e.preventDefault()
    if (!newJobInfo.employeeId || !newJobInfo.designation || !newJobInfo.department) {
      toast({
        title: "Error",
        description: "Employee ID, Designation, and Department are required",
        variant: "destructive",
      })
      return
    }

    const data: JobInfo = {
      id: Date.now().toString(),
      employeeId: newJobInfo.employeeId || "",
      designation: newJobInfo.designation || "",
      department: newJobInfo.department || "",
      reportingManager: newJobInfo.reportingManager || "",
      location: newJobInfo.location || "",
      workSchedule: newJobInfo.workSchedule || "",
      shiftTimings: newJobInfo.shiftTimings || "",
    }

    setJobInfoData((prev) => [...prev, data])
    setNewJobInfo({})
    toast({ title: "Success", description: "Job information added successfully" })
  }

  const handleDeleteJobInfo = (id: string) => {
    setJobInfoData((prev) => prev.filter((item) => item.id !== id))
    toast({ title: "Success", description: "Job information deleted" })
  }

  const handleCreateAttendanceLeave = (e: React.FormEvent) => {
    e.preventDefault()
    if (!newAttendanceLeave.employeeId) {
      toast({ title: "Error", description: "Employee ID is required", variant: "destructive" })
      return
    }

    const data: AttendanceLeave = {
      id: Date.now().toString(),
      employeeId: newAttendanceLeave.employeeId || "",
      casualLeave: newAttendanceLeave.casualLeave || 0,
      sickLeave: newAttendanceLeave.sickLeave || 0,
      earnedLeave: newAttendanceLeave.earnedLeave || 0,
      leaveHistory: newAttendanceLeave.leaveHistory || "",
      attendanceRecord: newAttendanceLeave.attendanceRecord || "",
      holidayCalendar: newAttendanceLeave.holidayCalendar || "",
    }

    setAttendanceLeaveData((prev) => [...prev, data])
    setNewAttendanceLeave({})
    toast({ title: "Success", description: "Attendance & Leave data added successfully" })
  }

  const handleDeleteAttendanceLeave = (id: string) => {
    setAttendanceLeaveData((prev) => prev.filter((item) => item.id !== id))
    toast({ title: "Success", description: "Attendance & Leave data deleted" })
  }

  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      <div className="max-w-7xl mx-auto p-6">
        <Tabs defaultValue="users" className="space-y-6">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="users" className="flex items-center gap-2">
              <Users className="h-4 w-4" />
              User Management
            </TabsTrigger>
            <TabsTrigger value="employee-data" className="flex items-center gap-2">
              <Database className="h-4 w-4" />
              Employee Data
            </TabsTrigger>
          </TabsList>

          {/* User Management Tab */}
          <TabsContent value="users" className="space-y-6">
            <UserManagement />
          </TabsContent>

          {/* Employee Data Management Tab */}
          <TabsContent value="employee-data" className="space-y-6">
            <Tabs defaultValue="personal" className="space-y-4">
              <TabsList className="grid w-full grid-cols-7">
                <TabsTrigger value="personal" className="flex items-center gap-1 text-xs">
                  <Users className="h-3 w-3" />
                  Personal
                </TabsTrigger>
                <TabsTrigger value="job" className="flex items-center gap-1 text-xs">
                  <Briefcase className="h-3 w-3" />
                  Job
                </TabsTrigger>
                <TabsTrigger value="attendance" className="flex items-center gap-1 text-xs">
                  <Calendar className="h-3 w-3" />
                  Attendance
                </TabsTrigger>
                <TabsTrigger value="payroll" className="flex items-center gap-1 text-xs">
                  <DollarSign className="h-3 w-3" />
                  Payroll
                </TabsTrigger>
                <TabsTrigger value="performance" className="flex items-center gap-1 text-xs">
                  <TrendingUp className="h-3 w-3" />
                  Performance
                </TabsTrigger>
                <TabsTrigger value="assets" className="flex items-center gap-1 text-xs">
                  <Building className="h-3 w-3" />
                  Assets
                </TabsTrigger>
                <TabsTrigger value="policies" className="flex items-center gap-1 text-xs">
                  <FileText className="h-3 w-3" />
                  Policies
                </TabsTrigger>
              </TabsList>

              {/* Personal Information Tab */}
              <TabsContent value="personal" className="space-y-6">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Plus className="h-5 w-5" />
                        Add Personal Information
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <form onSubmit={handleCreatePersonalInfo} className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-2">
                            <Label>Employee ID</Label>
                            <Input
                              placeholder="EMP001"
                              value={newPersonalInfo.employeeId || ""}
                              onChange={(e) => setNewPersonalInfo((prev) => ({ ...prev, employeeId: e.target.value }))}
                              required
                            />
                          </div>
                          <div className="space-y-2">
                            <Label>Full Name</Label>
                            <Input
                              placeholder="John Doe"
                              value={newPersonalInfo.name || ""}
                              onChange={(e) => setNewPersonalInfo((prev) => ({ ...prev, name: e.target.value }))}
                              required
                            />
                          </div>
                        </div>
                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-2">
                            <Label>Email</Label>
                            <Input
                              type="email"
                              placeholder="john@company.com"
                              value={newPersonalInfo.email || ""}
                              onChange={(e) => setNewPersonalInfo((prev) => ({ ...prev, email: e.target.value }))}
                              required
                            />
                          </div>
                          <div className="space-y-2">
                            <Label>Phone</Label>
                            <Input
                              placeholder="+1-555-0123"
                              value={newPersonalInfo.phone || ""}
                              onChange={(e) => setNewPersonalInfo((prev) => ({ ...prev, phone: e.target.value }))}
                            />
                          </div>
                        </div>
                        <div className="space-y-2">
                          <Label>Address</Label>
                          <Textarea
                            placeholder="123 Main St, City, State 12345"
                            value={newPersonalInfo.address || ""}
                            onChange={(e) => setNewPersonalInfo((prev) => ({ ...prev, address: e.target.value }))}
                            rows={2}
                          />
                        </div>
                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-2">
                            <Label>Emergency Contact</Label>
                            <Input
                              placeholder="Jane Doe"
                              value={newPersonalInfo.emergencyContact || ""}
                              onChange={(e) =>
                                setNewPersonalInfo((prev) => ({ ...prev, emergencyContact: e.target.value }))
                              }
                            />
                          </div>
                          <div className="space-y-2">
                            <Label>Emergency Phone</Label>
                            <Input
                              placeholder="+1-555-0124"
                              value={newPersonalInfo.emergencyPhone || ""}
                              onChange={(e) =>
                                setNewPersonalInfo((prev) => ({ ...prev, emergencyPhone: e.target.value }))
                              }
                            />
                          </div>
                        </div>
                        <div className="grid grid-cols-3 gap-4">
                          <div className="space-y-2">
                            <Label>Date of Birth</Label>
                            <Input
                              type="date"
                              value={newPersonalInfo.dateOfBirth || ""}
                              onChange={(e) => setNewPersonalInfo((prev) => ({ ...prev, dateOfBirth: e.target.value }))}
                            />
                          </div>
                          <div className="space-y-2">
                            <Label>Gender</Label>
                            <Select
                              value={newPersonalInfo.gender || ""}
                              onValueChange={(value) => setNewPersonalInfo((prev) => ({ ...prev, gender: value }))}
                            >
                              <SelectTrigger>
                                <SelectValue placeholder="Select" />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="Male">Male</SelectItem>
                                <SelectItem value="Female">Female</SelectItem>
                                <SelectItem value="Other">Other</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                          <div className="space-y-2">
                            <Label>Date of Joining</Label>
                            <Input
                              type="date"
                              value={newPersonalInfo.dateOfJoining || ""}
                              onChange={(e) =>
                                setNewPersonalInfo((prev) => ({ ...prev, dateOfJoining: e.target.value }))
                              }
                            />
                          </div>
                        </div>
                        <Button type="submit" className="w-full">
                          Add Personal Information
                        </Button>
                      </form>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle>Personal Information Records ({personalInfoData.length})</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3 max-h-96 overflow-y-auto">
                        {personalInfoData.map((info) => (
                          <div key={info.id} className="p-3 border border-border rounded-lg">
                            <div className="flex items-start justify-between mb-2">
                              <div className="flex-1">
                                <h4 className="font-medium text-sm">{info.name}</h4>
                                <p className="text-xs text-muted-foreground">ID: {info.employeeId}</p>
                                <p className="text-xs text-muted-foreground">{info.email}</p>
                              </div>
                              <Button variant="destructive" size="sm" onClick={() => handleDeletePersonalInfo(info.id)}>
                                <Trash2 className="h-4 w-4" />
                              </Button>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              {/* Job Information Tab */}
              <TabsContent value="job" className="space-y-6">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Plus className="h-5 w-5" />
                        Add Job Information
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <form onSubmit={handleCreateJobInfo} className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-2">
                            <Label>Employee ID</Label>
                            <Input
                              placeholder="EMP001"
                              value={newJobInfo.employeeId || ""}
                              onChange={(e) => setNewJobInfo((prev) => ({ ...prev, employeeId: e.target.value }))}
                              required
                            />
                          </div>
                          <div className="space-y-2">
                            <Label>Designation</Label>
                            <Input
                              placeholder="Software Engineer"
                              value={newJobInfo.designation || ""}
                              onChange={(e) => setNewJobInfo((prev) => ({ ...prev, designation: e.target.value }))}
                              required
                            />
                          </div>
                        </div>
                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-2">
                            <Label>Department</Label>
                            <Input
                              placeholder="Engineering"
                              value={newJobInfo.department || ""}
                              onChange={(e) => setNewJobInfo((prev) => ({ ...prev, department: e.target.value }))}
                              required
                            />
                          </div>
                          <div className="space-y-2">
                            <Label>Reporting Manager</Label>
                            <Input
                              placeholder="Jane Smith"
                              value={newJobInfo.reportingManager || ""}
                              onChange={(e) => setNewJobInfo((prev) => ({ ...prev, reportingManager: e.target.value }))}
                            />
                          </div>
                        </div>
                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-2">
                            <Label>Location</Label>
                            <Input
                              placeholder="New York Office"
                              value={newJobInfo.location || ""}
                              onChange={(e) => setNewJobInfo((prev) => ({ ...prev, location: e.target.value }))}
                            />
                          </div>
                          <div className="space-y-2">
                            <Label>Work Schedule</Label>
                            <Select
                              value={newJobInfo.workSchedule || ""}
                              onValueChange={(value) => setNewJobInfo((prev) => ({ ...prev, workSchedule: value }))}
                            >
                              <SelectTrigger>
                                <SelectValue placeholder="Select" />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="Full-time">Full-time</SelectItem>
                                <SelectItem value="Part-time">Part-time</SelectItem>
                                <SelectItem value="Contract">Contract</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                        </div>
                        <div className="space-y-2">
                          <Label>Shift Timings</Label>
                          <Input
                            placeholder="9:00 AM - 6:00 PM"
                            value={newJobInfo.shiftTimings || ""}
                            onChange={(e) => setNewJobInfo((prev) => ({ ...prev, shiftTimings: e.target.value }))}
                          />
                        </div>
                        <Button type="submit" className="w-full">
                          Add Job Information
                        </Button>
                      </form>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle>Job Information Records ({jobInfoData.length})</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3 max-h-96 overflow-y-auto">
                        {jobInfoData.map((job) => (
                          <div key={job.id} className="p-3 border border-border rounded-lg">
                            <div className="flex items-start justify-between mb-2">
                              <div className="flex-1">
                                <h4 className="font-medium text-sm">{job.designation}</h4>
                                <p className="text-xs text-muted-foreground">ID: {job.employeeId}</p>
                                <p className="text-xs text-muted-foreground">
                                  {job.department} • {job.location}
                                </p>
                              </div>
                              <Button variant="destructive" size="sm" onClick={() => handleDeleteJobInfo(job.id)}>
                                <Trash2 className="h-4 w-4" />
                              </Button>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              {/* Attendance & Leave Tab */}
              <TabsContent value="attendance" className="space-y-6">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Plus className="h-5 w-5" />
                        Add Attendance & Leave Data
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <form onSubmit={handleCreateAttendanceLeave} className="space-y-4">
                        <div className="space-y-2">
                          <Label>Employee ID</Label>
                          <Input
                            placeholder="EMP001"
                            value={newAttendanceLeave.employeeId || ""}
                            onChange={(e) => setNewAttendanceLeave((prev) => ({ ...prev, employeeId: e.target.value }))}
                            required
                          />
                        </div>
                        <div className="grid grid-cols-3 gap-4">
                          <div className="space-y-2">
                            <Label>Casual Leave</Label>
                            <Input
                              type="number"
                              placeholder="12"
                              value={newAttendanceLeave.casualLeave || ""}
                              onChange={(e) =>
                                setNewAttendanceLeave((prev) => ({
                                  ...prev,
                                  casualLeave: Number.parseInt(e.target.value) || 0,
                                }))
                              }
                            />
                          </div>
                          <div className="space-y-2">
                            <Label>Sick Leave</Label>
                            <Input
                              type="number"
                              placeholder="10"
                              value={newAttendanceLeave.sickLeave || ""}
                              onChange={(e) =>
                                setNewAttendanceLeave((prev) => ({
                                  ...prev,
                                  sickLeave: Number.parseInt(e.target.value) || 0,
                                }))
                              }
                            />
                          </div>
                          <div className="space-y-2">
                            <Label>Earned Leave</Label>
                            <Input
                              type="number"
                              placeholder="15"
                              value={newAttendanceLeave.earnedLeave || ""}
                              onChange={(e) =>
                                setNewAttendanceLeave((prev) => ({
                                  ...prev,
                                  earnedLeave: Number.parseInt(e.target.value) || 0,
                                }))
                              }
                            />
                          </div>
                        </div>
                        <div className="space-y-2">
                          <Label>Leave History</Label>
                          <Textarea
                            placeholder="Recent leave applications and approvals..."
                            value={newAttendanceLeave.leaveHistory || ""}
                            onChange={(e) =>
                              setNewAttendanceLeave((prev) => ({ ...prev, leaveHistory: e.target.value }))
                            }
                            rows={2}
                          />
                        </div>
                        <div className="space-y-2">
                          <Label>Attendance Record</Label>
                          <Input
                            placeholder="98% attendance rate"
                            value={newAttendanceLeave.attendanceRecord || ""}
                            onChange={(e) =>
                              setNewAttendanceLeave((prev) => ({ ...prev, attendanceRecord: e.target.value }))
                            }
                          />
                        </div>
                        <Button type="submit" className="w-full">
                          Add Attendance & Leave Data
                        </Button>
                      </form>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle>Attendance & Leave Records ({attendanceLeaveData.length})</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3 max-h-96 overflow-y-auto">
                        {attendanceLeaveData.map((attendance) => (
                          <div key={attendance.id} className="p-3 border border-border rounded-lg">
                            <div className="flex items-start justify-between mb-2">
                              <div className="flex-1">
                                <h4 className="font-medium text-sm">Employee: {attendance.employeeId}</h4>
                                <div className="flex gap-2 mt-1">
                                  <Badge variant="outline">CL: {attendance.casualLeave}</Badge>
                                  <Badge variant="outline">SL: {attendance.sickLeave}</Badge>
                                  <Badge variant="outline">EL: {attendance.earnedLeave}</Badge>
                                </div>
                              </div>
                              <Button
                                variant="destructive"
                                size="sm"
                                onClick={() => handleDeleteAttendanceLeave(attendance.id)}
                              >
                                <Trash2 className="h-4 w-4" />
                              </Button>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              {/* Placeholder tabs for remaining data types */}
              <TabsContent value="payroll" className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Payroll & Compensation Management</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground">
                      Payroll management interface - salary, bonuses, deductions, tax information
                    </p>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="performance" className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Performance & Appraisal Management</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground">Performance reviews, goals, achievements, training records</p>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="assets" className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Administrative & Asset Management</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground">
                      ID cards, laptop assignments, seating, building access, travel
                    </p>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="policies" className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle>HR Policies & Compliance Management</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground">
                      Company policies, code of conduct, compliance forms, certifications
                    </p>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
