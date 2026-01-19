'use client';

import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from '@/components/ai-elements/conversation';
import {
  Message,
  MessageContent,
  MessageResponse,
  MessageActions,
  MessageAction,
} from '@/components/ai-elements/message';
import {
  PromptInput,
  PromptInputActionAddAttachments,
  PromptInputActionMenu,
  PromptInputActionMenuContent,
  PromptInputActionMenuTrigger,
  PromptInputAttachment,
  PromptInputAttachments,
  PromptInputBody,
  PromptInputButton,
  PromptInputHeader,
  type PromptInputMessage,
  PromptInputSelect,
  PromptInputSelectContent,
  PromptInputSelectItem,
  PromptInputSelectTrigger,
  PromptInputSelectValue,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputFooter,
  PromptInputTools,
} from '@/components/ai-elements/prompt-input';
import { useState, useEffect } from 'react';
import { CopyIcon, GlobeIcon, RefreshCcwIcon, PlusIcon, TrashIcon, MessageSquareIcon, GraduationCapIcon } from 'lucide-react';
import {
  Source,
  Sources,
  SourcesContent,
  SourcesTrigger,
} from '@/components/ai-elements/sources';
import {
  Reasoning,
  ReasoningContent,
  ReasoningTrigger,
} from '@/components/ai-elements/reasoning';


const programs = [
  { name: 'Alle Studiengänge', value: 'all' },
  { name: 'Informatik', value: 'Informatik' },
  { name: 'BWL', value: 'Betriebswirtschaftslehre' },
  { name: 'Technische Informatik', value: 'Technische Informatik' },
  { name: 'Wirtschaftsinformatik', value: 'Wirtschaftsinformatik' },
];

// Programs for the welcome screen (excluding "Alle Studiengänge")
const selectablePrograms = programs.filter(p => p.value !== 'all');

type MessagePart = {
  type: 'text' | 'reasoning' | 'source-url';
  text: string;
  url?: string;
};

type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  parts: MessagePart[];
};

type Chat = {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: Date;
};

export function ChatTab() {
  const [input, setInput] = useState('');
  const [program, setProgram] = useState<string>('all');
  const [webSearch, setWebSearch] = useState(false);
  const [chats, setChats] = useState<Chat[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [isLoadingChats, setIsLoadingChats] = useState(true);

  const currentChat = chats.find((chat) => chat.id === currentChatId);
  const messages = currentChat?.messages || [];

  // Load chats from MongoDB on mount
  useEffect(() => {
    const loadChats = async () => {
      try {
        const response = await fetch('/api/chats');
        if (response.ok) {
          const data = await response.json();
          setChats(data.chats.map((chat: any) => ({
            ...chat,
            createdAt: new Date(chat.createdAt),
          })));
        }
      } catch (error) {
        console.error('Failed to load chats:', error);
      } finally {
        setIsLoadingChats(false);
      }
    };
    loadChats();
  }, []);

  // Helper function to save chat to MongoDB
  const saveChatToMongoDB = async (chat: Chat) => {
    try {
      await fetch('/api/chats', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(chat),
      });
    } catch (error) {
      console.error('Failed to save chat to MongoDB:', error);
    }
  };

  const createNewChat = async () => {
    const newChat: Chat = {
      id: `chat-${Date.now()}`,
      title: 'New Chat',
      messages: [],
      createdAt: new Date(),
    };
    setChats((prev) => [newChat, ...prev]);
    setCurrentChatId(newChat.id);
    await saveChatToMongoDB(newChat);
    return newChat.id;
  };

  const handleSubmit = async (message: PromptInputMessage) => {
    const hasText = Boolean(message.text);
    const hasAttachments = Boolean(message.files?.length);

    if (!(hasText || hasAttachments)) {
      return;
    }

    // Create new chat if none exists
    let chatId = currentChatId;
    if (!chatId) {
      chatId = await createNewChat();
    }

    // Create user message
    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      parts: [
        {
          type: 'text',
          text: message.text || 'Sent with attachments',
        },
      ],
    };

    // Create temporary assistant response with loading state
    const assistantMessageId = `assistant-${Date.now()}`;
    const loadingMessage: ChatMessage = {
      id: assistantMessageId,
      role: 'assistant',
      parts: [
        {
          type: 'text',
          text: 'Thinking...',
        },
      ],
    };

    // Add user message and loading message to chat
    let updatedChat: Chat | null = null;
    setChats((prev) =>
      prev.map((chat) => {
        if (chat.id === chatId) {
          const updatedMessages = [...chat.messages, userMessage, loadingMessage];
          // Update chat title with first message if it's still "New Chat"
          const updatedTitle = chat.title === 'New Chat' && message.text
            ? message.text.slice(0, 50) + (message.text.length > 50 ? '...' : '')
            : chat.title;
          updatedChat = { ...chat, messages: updatedMessages, title: updatedTitle };
          return updatedChat;
        }
        return chat;
      })
    );
    setInput('');

    // Save to MongoDB after adding user message
    if (updatedChat) {
      await saveChatToMongoDB(updatedChat);
    }

    try {
      // Call the RAG API
      const ragApiUrl = process.env.NEXT_PUBLIC_RAG_API_URL || 'http://localhost:8000';
      const response = await fetch(`${ragApiUrl}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: message.text,
          session_id: chatId,
          context: program && program !== 'all' ? { program: [program], degree: [], doctype: [], status: [] } : undefined,
        }),
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      const responseData = await response.json();
      const responseText = responseData.answer;

      // Update the assistant message with the actual response
      let finalChat: Chat | null = null;
      setChats((prev) =>
        prev.map((chat) => {
          if (chat.id === chatId) {
            finalChat = {
              ...chat,
              messages: chat.messages.map((msg) =>
                msg.id === assistantMessageId
                  ? {
                      ...msg,
                      parts: [
                        {
                          type: 'text',
                          text: responseText,
                        },
                      ],
                    }
                  : msg
              ),
            };
            return finalChat;
          }
          return chat;
        })
      );

      // Save to MongoDB after receiving assistant response
      if (finalChat) {
        await saveChatToMongoDB(finalChat);
      }
    } catch (error) {
      console.error('Error calling search API:', error);
      // Update the assistant message with error
      let errorChat: Chat | null = null;
      setChats((prev) =>
        prev.map((chat) => {
          if (chat.id === chatId) {
            errorChat = {
              ...chat,
              messages: chat.messages.map((msg) =>
                msg.id === assistantMessageId
                  ? {
                      ...msg,
                      parts: [
                        {
                          type: 'text',
                          text: `Error: Failed to get response from the server. ${error instanceof Error ? error.message : 'Unknown error'}`,
                        },
                      ],
                    }
                  : msg
              ),
            };
            return errorChat;
          }
          return chat;
        })
      );

      // Save to MongoDB even in error case
      if (errorChat) {
        await saveChatToMongoDB(errorChat);
      }
    }
  };

  const regenerate = () => {
    if (!currentChatId) return;

    // Find the last assistant message and regenerate it
    const lastAssistantIndex = messages.findLastIndex((m) => m.role === 'assistant');
    if (lastAssistantIndex === -1) return;

    setChats((prev) =>
      prev.map((chat) => {
        if (chat.id === currentChatId) {
          const updatedMessages = [...chat.messages];
          const lastAssistantMessage = updatedMessages[lastAssistantIndex];

          // Simple regeneration - just add a timestamp
          lastAssistantMessage.parts = lastAssistantMessage.parts.map((part) => ({
            ...part,
            text: `${part.text} (regenerated at ${new Date().toLocaleTimeString()})`,
          }));

          return { ...chat, messages: updatedMessages };
        }
        return chat;
      })
    );
  };

  const deleteChat = async (chatId: string) => {
    setChats((prev) => prev.filter((chat) => chat.id !== chatId));
    if (currentChatId === chatId) {
      setCurrentChatId(null);
    }

    // Delete from MongoDB
    try {
      await fetch(`/api/chats?id=${chatId}`, {
        method: 'DELETE',
      });
    } catch (error) {
      console.error('Failed to delete chat from MongoDB:', error);
    }
  };

  return (
    <div className="flex h-full gap-4">
      {/* Sidebar */}
      <div className="w-64 flex-shrink-0 border-r border-border flex flex-col">
        <div className="p-3 border-b border-border">
          <button
            onClick={createNewChat}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors"
          >
            <PlusIcon className="size-4" />
            <span>New Chat</span>
          </button>
        </div>
        <div className="flex-1 overflow-y-auto">
          {chats.length === 0 ? (
            <div className="p-4 text-center text-muted-foreground text-sm">
              No chats yet. Start a new chat!
            </div>
          ) : (
            <div className="p-2 space-y-1">
              {chats.map((chat) => (
                <div
                  key={chat.id}
                  className={`group flex items-center gap-2 p-3 rounded-md cursor-pointer transition-colors ${
                    currentChatId === chat.id
                      ? 'bg-accent text-accent-foreground'
                      : 'hover:bg-accent/50'
                  }`}
                >
                  <MessageSquareIcon className="size-4 flex-shrink-0" />
                  <button
                    onClick={() => setCurrentChatId(chat.id)}
                    className="flex-1 text-left text-sm truncate"
                    title={chat.title}
                  >
                    {chat.title}
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteChat(chat.id);
                    }}
                    className="opacity-0 group-hover:opacity-100 p-1 hover:bg-destructive/10 rounded transition-opacity"
                    title="Delete chat"
                  >
                    <TrashIcon className="size-3 text-destructive" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="flex-1 overflow-hidden">
          <Conversation className="h-full overflow-x-hidden">
          <ConversationContent>
            {/* Welcome screen when no messages */}
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full text-center px-4">
                <GraduationCapIcon className="size-16 text-primary mb-6" />
                <h2 className="text-2xl font-semibold mb-3">Willkommen beim FH Wedel Chat Bot</h2>
                <p className="text-muted-foreground mb-6 max-w-md">
                  Ich kann dir bei Fragen zu den Studiengängen der FH Wedel helfen.
                  Wähle einen Studiengang aus oder stelle eine allgemeine Frage.
                </p>
                <div className="flex flex-wrap gap-3 justify-center max-w-lg">
                  {selectablePrograms.map((prog) => (
                    <button
                      key={prog.value}
                      onClick={() => setProgram(prog.value)}
                      className={`px-4 py-2 rounded-full border transition-colors ${
                        program === prog.value
                          ? 'bg-primary text-primary-foreground border-primary'
                          : 'bg-background hover:bg-accent border-border hover:border-primary'
                      }`}
                    >
                      {prog.name}
                    </button>
                  ))}
                </div>
                {program !== 'all' && (
                  <p className="mt-4 text-sm text-primary">
                    Filter aktiv: <span className="font-medium">{programs.find(p => p.value === program)?.name}</span>
                  </p>
                )}
                <p className="mt-6 text-sm text-muted-foreground">
                  Stelle deine Frage unten ein, um loszulegen.
                </p>
              </div>
            )}
            {messages.map((message) => (
              <div key={message.id}>
                {message.role === 'assistant' && message.parts.filter((part) => part.type === 'source-url').length > 0 && (
                  <Sources>
                    <SourcesTrigger
                      count={
                        message.parts.filter(
                          (part) => part.type === 'source-url',
                        ).length
                      }
                    />
                    {message.parts.filter((part) => part.type === 'source-url').map((part, i) => (
                      <SourcesContent key={`${message.id}-${i}`}>
                        <Source
                          key={`${message.id}-${i}`}
                          href={part.url}
                          title={part.url}
                        />
                      </SourcesContent>
                    ))}
                  </Sources>
                )}
                {message.parts.map((part, i) => {
                  switch (part.type) {
                    case 'text':
                      return (
                        <Message key={`${message.id}-${i}`} from={message.role}>
                          <MessageContent className="max-w-full">
                            <MessageResponse className="break-words overflow-wrap-anywhere whitespace-pre-wrap chat-message-text">
                              {part.text}
                            </MessageResponse>
                          </MessageContent>
                          {message.role === 'assistant' && i === messages.length - 1 && (
                            <MessageActions>
                              <MessageAction
                                onClick={() => regenerate()}
                                label="Retry"
                              >
                                <RefreshCcwIcon className="size-3" />
                              </MessageAction>
                              <MessageAction
                                onClick={() =>
                                  navigator.clipboard.writeText(part.text)
                                }
                                label="Copy"
                              >
                                <CopyIcon className="size-3" />
                              </MessageAction>
                            </MessageActions>
                          )}
                        </Message>
                      );
                    case 'reasoning':
                      return (
                        <Reasoning
                          key={`${message.id}-${i}`}
                          className="w-full"
                          isStreaming={false}
                        >
                          <ReasoningTrigger />
                          <ReasoningContent>{part.text}</ReasoningContent>
                        </Reasoning>
                      );
                    default:
                      return null;
                  }
                })}
              </div>
            ))}
          </ConversationContent>
          <ConversationScrollButton />
        </Conversation>
        </div>

        <div className="flex-shrink-0 mt-4">
          <PromptInput onSubmit={handleSubmit} globalDrop multiple>
            <PromptInputHeader>
              <PromptInputAttachments>
                {(attachment) => <PromptInputAttachment data={attachment} />}
              </PromptInputAttachments>
            </PromptInputHeader>
            <PromptInputBody>
              <PromptInputTextarea
                onChange={(e) => setInput(e.target.value)}
                value={input}
                className="chat-input-text"
              />
            </PromptInputBody>
            <PromptInputFooter>
              <PromptInputTools>
                <PromptInputActionMenu>
                  <PromptInputActionMenuTrigger />
                  <PromptInputActionMenuContent>
                    <PromptInputActionAddAttachments />
                  </PromptInputActionMenuContent>
                </PromptInputActionMenu>
                <PromptInputButton
                  variant={webSearch ? 'default' : 'ghost'}
                  onClick={() => setWebSearch(!webSearch)}
                >
                  <GlobeIcon size={16} />
                  <span>Search</span>
                </PromptInputButton>
                <PromptInputSelect
                  onValueChange={(value) => {
                    setProgram(value);
                  }}
                  value={program}
                >
                  <PromptInputSelectTrigger>
                    <PromptInputSelectValue placeholder="Studiengang" />
                  </PromptInputSelectTrigger>
                  <PromptInputSelectContent>
                    {programs.map((prog) => (
                      <PromptInputSelectItem key={prog.value} value={prog.value}>
                        {prog.name}
                      </PromptInputSelectItem>
                    ))}
                  </PromptInputSelectContent>
                </PromptInputSelect>
              </PromptInputTools>
              <PromptInputSubmit disabled={!input} />
            </PromptInputFooter>
          </PromptInput>
        </div>
      </div>
    </div>
  );
};

export default ChatTab;