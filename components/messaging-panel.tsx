"use client"

import { useState, useEffect, useRef } from "react"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Send, Search } from "lucide-react"

interface Message {
  id: string
  senderId: string
  receiverId: string
  content: string
  timestamp: string
  isRead: boolean
}

interface Conversation {
  id: string
  user: {
    id: string
    username: string
    profilePic: string
    isOnline: boolean
  }
  lastMessage: string
  timestamp: string
  unreadCount: number
}

interface MessagingPanelProps {
  onClose: () => void
}

export function MessagingPanel({ onClose }: MessagingPanelProps) {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [selectedConversation, setSelectedConversation] = useState<Conversation | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [newMessage, setNewMessage] = useState("")
  const [searchQuery, setSearchQuery] = useState("")
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    loadConversations()
  }, [])

  useEffect(() => {
    if (selectedConversation) {
      loadMessages(selectedConversation.id)
    }
  }, [selectedConversation])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const loadConversations = () => {
    const mockConversations: Conversation[] = [
      {
        id: "1",
        user: {
          id: "1",
          username: "alice_doe",
          profilePic: "/placeholder.svg?height=40&width=40",
          isOnline: true,
        },
        lastMessage: "Hey! How are you doing?",
        timestamp: "2m",
        unreadCount: 2,
      },
      {
        id: "2",
        user: {
          id: "2",
          username: "john_smith",
          profilePic: "/placeholder.svg?height=40&width=40",
          isOnline: false,
        },
        lastMessage: "Thanks for sharing that recipe!",
        timestamp: "1h",
        unreadCount: 0,
      },
    ]
    setConversations(mockConversations)
  }

  const loadMessages = (conversationId: string) => {
    const mockMessages: Message[] = [
      {
        id: "1",
        senderId: "1",
        receiverId: "current",
        content: "Hey! How are you doing?",
        timestamp: "10:30 AM",
        isRead: true,
      },
      {
        id: "2",
        senderId: "current",
        receiverId: "1",
        content: "I'm doing great! Just posted a new photo.",
        timestamp: "10:32 AM",
        isRead: true,
      },
      {
        id: "3",
        senderId: "1",
        receiverId: "current",
        content: "I saw it! Beautiful sunset 🌅",
        timestamp: "10:33 AM",
        isRead: false,
      },
    ]
    setMessages(mockMessages)
  }

  const sendMessage = () => {
    if (!newMessage.trim() || !selectedConversation) return

    const message: Message = {
      id: `msg_${Math.random().toString(36).substr(2, 9)}`,
      senderId: "current",
      receiverId: selectedConversation.user.id,
      content: newMessage,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      isRead: false,
    }

    setMessages([...messages, message])
    setNewMessage("")

    // Update conversation
    setConversations(
      conversations.map((conv) =>
        conv.id === selectedConversation.id ? { ...conv, lastMessage: newMessage, timestamp: "now" } : conv,
      ),
    )
  }

  const filteredConversations = conversations.filter((conv) =>
    conv.user.username.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  return (
    <Dialog open={true} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl h-[600px] p-0">
        <div className="flex h-full">
          {/* Conversations List */}
          <div className="w-1/3 border-r border-gray-200">
            <DialogHeader className="p-4 border-b">
              <DialogTitle>Messages</DialogTitle>
            </DialogHeader>

            <div className="p-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                <Input
                  placeholder="Search conversations..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>

            <ScrollArea className="flex-1">
              {filteredConversations.map((conversation) => (
                <div
                  key={conversation.id}
                  className={`p-4 cursor-pointer hover:bg-gray-50 border-b ${
                    selectedConversation?.id === conversation.id ? "bg-blue-50" : ""
                  }`}
                  onClick={() => setSelectedConversation(conversation)}
                >
                  <div className="flex items-center gap-3">
                    <div className="relative">
                      <Avatar>
                        <AvatarImage src={conversation.user.profilePic || "/placeholder.svg"} />
                        <AvatarFallback>{conversation.user.username[0].toUpperCase()}</AvatarFallback>
                      </Avatar>
                      {conversation.user.isOnline && (
                        <div className="absolute bottom-0 right-0 w-3 h-3 bg-green-500 rounded-full border-2 border-white" />
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <p className="font-semibold truncate">{conversation.user.username}</p>
                        <span className="text-xs text-gray-500">{conversation.timestamp}</span>
                      </div>
                      <p className="text-sm text-gray-600 truncate">{conversation.lastMessage}</p>
                    </div>
                    {conversation.unreadCount > 0 && (
                      <div className="bg-blue-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                        {conversation.unreadCount}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </ScrollArea>
          </div>

          {/* Chat Area */}
          <div className="flex-1 flex flex-col">
            {selectedConversation ? (
              <>
                {/* Chat Header */}
                <div className="p-4 border-b flex items-center gap-3">
                  <Avatar>
                    <AvatarImage src={selectedConversation.user.profilePic || "/placeholder.svg"} />
                    <AvatarFallback>{selectedConversation.user.username[0].toUpperCase()}</AvatarFallback>
                  </Avatar>
                  <div>
                    <p className="font-semibold">{selectedConversation.user.username}</p>
                    <p className="text-sm text-gray-500">
                      {selectedConversation.user.isOnline ? "Active now" : "Last seen recently"}
                    </p>
                  </div>
                </div>

                {/* Messages */}
                <ScrollArea className="flex-1 p-4">
                  <div className="space-y-4">
                    {messages.map((message) => (
                      <div
                        key={message.id}
                        className={`flex ${message.senderId === "current" ? "justify-end" : "justify-start"}`}
                      >
                        <div
                          className={`max-w-xs px-4 py-2 rounded-lg ${
                            message.senderId === "current" ? "bg-blue-500 text-white" : "bg-gray-200 text-gray-900"
                          }`}
                        >
                          <p>{message.content}</p>
                          <p
                            className={`text-xs mt-1 ${
                              message.senderId === "current" ? "text-blue-100" : "text-gray-500"
                            }`}
                          >
                            {message.timestamp}
                          </p>
                        </div>
                      </div>
                    ))}
                    <div ref={messagesEndRef} />
                  </div>
                </ScrollArea>

                {/* Message Input */}
                <div className="p-4 border-t">
                  <div className="flex gap-2">
                    <Input
                      placeholder="Type a message..."
                      value={newMessage}
                      onChange={(e) => setNewMessage(e.target.value)}
                      onKeyPress={(e) => e.key === "Enter" && sendMessage()}
                      className="flex-1"
                    />
                    <Button onClick={sendMessage} disabled={!newMessage.trim()}>
                      <Send className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </>
            ) : (
              <div className="flex-1 flex items-center justify-center text-gray-500">
                Select a conversation to start messaging
              </div>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
