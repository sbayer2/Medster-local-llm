'use client'

import { Message } from '@/lib/types'
import { useEffect, useRef } from 'react'

interface MessageListProps {
    messages: Message[]
}

export default function MessageList({ messages }: MessageListProps) {
    const messagesEndRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages])

    const formatTime = (date: Date) => {
        return new Date(date).toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
        })
    }

    return (
        <div className="h-[500px] overflow-y-auto mb-6 space-y-4 px-2">
            {messages.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-center">
                    <div className="text-6xl mb-4">🏥</div>
                    <h3 className="text-xl font-semibold text-medical-400 mb-2">
                        Welcome to Medster Local LLM
                    </h3>
                    <p className="text-dark-text/60 max-w-md">
                        Ask me anything about patient data, lab results, medications, or clinical analysis.
                        All processing happens locally on your machine.
                    </p>
                </div>
            ) : (
                messages.map((message) => (
                    <div
                        key={message.id}
                        className={`message-enter ${message.role === 'user'
                                ? 'ml-auto'
                                : message.role === 'system'
                                    ? 'mx-auto'
                                    : ''
                            } ${message.role === 'user'
                                ? 'max-w-[80%]'
                                : message.role === 'system'
                                    ? 'max-w-[90%]'
                                    : 'max-w-[95%]'
                            }`}
                    >
                        <div
                            className={`rounded-xl p-4 ${message.role === 'user'
                                    ? 'bg-medical-gradient text-white'
                                    : message.role === 'system'
                                        ? 'bg-dark-surface/50 text-dark-text/70 text-sm border border-dark-border'
                                        : 'bg-dark-surface text-dark-text border border-dark-border'
                                }`}
                        >
                            <div className="flex items-start justify-between gap-3 mb-1">
                                <span className="font-semibold text-sm">
                                    {message.role === 'user'
                                        ? 'You'
                                        : message.role === 'system'
                                            ? 'System'
                                            : 'Medster'}
                                </span>
                                <span className="text-xs opacity-60">
                                    {formatTime(message.timestamp)}
                                </span>
                            </div>
                            <div className="whitespace-pre-wrap break-words">
                                {message.content}
                            </div>
                        </div>
                    </div>
                ))
            )}
            <div ref={messagesEndRef} />
        </div>
    )
}
