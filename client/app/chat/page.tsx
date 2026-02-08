import { AuthGuard } from "@/components/auth-guard"
import { ChatInterface } from "@/components/chat-interface"

export default function ChatPage() {
  return (
    <AuthGuard allowedRoles={["employee", "manager", "hr"]}>
      <ChatInterface />
    </AuthGuard>
  )
}
