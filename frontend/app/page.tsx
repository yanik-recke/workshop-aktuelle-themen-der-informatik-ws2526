import ChatWindow from "@/components/chat-window"

export default function Home() {
  return (
      <div className={`container mx-auto p-6 h-[calc(100vh-4rem)] overflow-hidden`}>
         <ChatWindow />
      </div>
  )
}
