"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { useAuth } from "@/components/auth-guard"
import { Navigation } from "@/components/navigation"
import { MessageContent } from "@/components/message-content"
import { Send, Bot, User } from "lucide-react"
import { apiClient } from "@/lib/api"
import { formatMultiAgentResponse, isDebuggingResponse, getDebugFallbackMessage } from "@/lib/message-formatter"

interface Message {
  id: string
  content: string
  sender: "user" | "bot"
  timestamp: Date
  agentUsed?: string
}

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      content:
        "Hello! I'm your **Multi-Agent HR Assistant**. I can help you with:\n\n• **Employee & Project Information** - Find details about team members, project assignments, and departmental data\n• **Company Policies** - Access information about HR policies, procedures, and guidelines\n• **Organizational Structure** - Learn about company structure, roles, and organizational information\n\nWhat would you like to know today?",
      sender: "bot",
      timestamp: new Date(),
      agentUsed: "Welcome Assistant"
    },
  ])
  const [inputMessage, setInputMessage] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState<string>("")
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { user } = useAuth()

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!inputMessage.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      sender: "user",
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    const currentMessage = inputMessage
    setInputMessage("")
    setIsLoading(true)

    try {
      // Call the multi-agent RAG API
      const response = await apiClient.multiAgentChat(currentMessage, sessionId)
      
      // Update session ID if it's new
      if (response.session_id && response.session_id !== sessionId) {
        setSessionId(response.session_id)
      }

      // Check if response looks like debugging output and format accordingly
      let finalContent = response.response;
      let finalAgent = response.agent_used;
      
      if (isDebuggingResponse(response.response)) {
        // If we detect debugging output, provide a cleaner fallback
        const queryType = currentMessage.toLowerCase().includes('policy') ? 'policy' :
                         currentMessage.toLowerCase().includes('employee') || currentMessage.toLowerCase().includes('department') ? 'employee' :
                         currentMessage.toLowerCase().includes('organization') || currentMessage.toLowerCase().includes('company') ? 'organization' : 'default';
        
        finalContent = getDebugFallbackMessage(queryType);
        finalAgent = 'System Processing';
      } else {
        // Format the response for better display
        const formatted = formatMultiAgentResponse(response.response, response.agent_used);
        finalContent = formatted.content;
        finalAgent = formatted.agentUsed || response.agent_used;
      }

      const botResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: finalContent,
        sender: "bot",
        timestamp: new Date(),
        agentUsed: finalAgent,
      }
      
      setMessages((prev) => [...prev, botResponse])
    } catch (error) {
      console.error('Chat error:', error)
      
      // Fallback error message
      const errorResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: "I'm sorry, I'm having trouble processing your request right now. Please try again in a moment. If the issue persists, please contact the system administrator.",
        sender: "bot",
        timestamp: new Date(),
      }
      
      setMessages((prev) => [...prev, errorResponse])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Navigation />

      {/* Chat Messages */}
      <div className="flex-1 overflow-hidden">
        <div className="max-w-4xl mx-auto h-full flex flex-col">
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-3 ${message.sender === "user" ? "justify-end" : "justify-start"}`}
              >
                {message.sender === "bot" && (
                  <Avatar className="h-8 w-8 bg-primary">
                    <AvatarFallback>
                      <Bot className="h-4 w-4 text-primary-foreground" />
                    </AvatarFallback>
                  </Avatar>
                )}

                <div
                  className={`max-w-[70%] rounded-lg p-3 ${
                    message.sender === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-card text-card-foreground border border-border"
                  }`}
                >
                  {message.sender === "user" ? (
                    <p className="text-sm leading-relaxed">{message.content}</p>
                  ) : (
                    <MessageContent content={message.content} />
                  )}
                  <div className="flex justify-between items-center mt-2">
                    <p className="text-xs opacity-70">{message.timestamp.toLocaleTimeString()}</p>
                    {message.sender === "bot" && message.agentUsed && (
                      <p className="text-xs opacity-60 italic">Agent: {message.agentUsed}</p>
                    )}
                  </div>
                </div>

                {message.sender === "user" && (
                  <Avatar className="h-8 w-8 bg-secondary">
                    <AvatarFallback>
                      <User className="h-4 w-4 text-secondary-foreground" />
                    </AvatarFallback>
                  </Avatar>
                )}
              </div>
            ))}

            {isLoading && (
              <div className="flex gap-3 justify-start">
                <Avatar className="h-8 w-8 bg-primary">
                  <AvatarFallback>
                    <Bot className="h-4 w-4 text-primary-foreground" />
                  </AvatarFallback>
                </Avatar>
                <div className="bg-card text-card-foreground border border-border rounded-lg p-3">
                  <div className="flex items-center gap-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>
                    <span className="text-sm text-muted-foreground">Thinking...</span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Message Input */}
          <div className="border-t border-border p-4">
            <form onSubmit={handleSendMessage} className="flex gap-2">
              <Input
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="Ask me about HR policies, benefits, leave, or any employee-related questions..."
                className="flex-1"
                disabled={isLoading}
              />
              <Button type="submit" disabled={isLoading || !inputMessage.trim()}>
                <Send className="h-4 w-4" />
              </Button>
            </form>
          </div>
        </div>
      </div>
    </div>
  )
}
