'use client'

import { useState, useEffect, useRef } from 'react'
import { Message, ModelInfo, AgentEvent, ConnectionStatus } from '@/lib/types'
import { MedsterWebSocket } from '@/lib/websocket'
import ModelSelector from './ModelSelector'
import MessageList from './MessageList'
import MessageInput from './MessageInput'
import StatusIndicator from './StatusIndicator'

export default function ChatInterface() {
    const [messages, setMessages] = useState<Message[]>([])
    const [status, setStatus] = useState<ConnectionStatus>({
        connected: false,
        model: null,
        processing: false,
    })
    const [currentActivity, setCurrentActivity] = useState<string>('')
    const [availableModels, setAvailableModels] = useState<ModelInfo[]>([])
    const wsRef = useRef<MedsterWebSocket | null>(null)

    useEffect(() => {
        // Initialize WebSocket connection
        const ws = new MedsterWebSocket()
        wsRef.current = ws

        ws.connect(
            handleWebSocketMessage,
            () => setStatus(prev => ({ ...prev, connected: true })),
            () => setStatus(prev => ({ ...prev, connected: false, processing: false })),
            (error) => console.error('WebSocket error:', error)
        )

        // Fetch current model and available models on mount
        fetchCurrentModel()
        fetchAvailableModels()

        return () => {
            ws.disconnect()
        }
    }, [])

    const fetchCurrentModel = async () => {
        try {
            const response = await fetch('http://localhost:8000/api/current-model')
            const data = await response.json()
            setStatus(prev => ({ ...prev, model: data.model }))
        } catch (error) {
            console.error('Error fetching current model:', error)
        }
    }

    const fetchAvailableModels = async () => {
        try {
            const response = await fetch('http://localhost:8000/api/models')
            const data = await response.json()
            setAvailableModels(data)
        } catch (error) {
            console.error('Error fetching available models:', error)
        }
    }

    const handleWebSocketMessage = (event: AgentEvent) => {
        switch (event.type) {
            case 'start':
                setStatus(prev => ({ ...prev, processing: true }))
                setCurrentActivity('Processing query...')
                break

            case 'task_start':
                setCurrentActivity(`Working on: ${event.data.task}`)
                addSystemMessage(`📋 Task: ${event.data.task}`)
                break

            case 'task_complete':
                addSystemMessage(`✅ Completed: ${event.data.task}`)
                break

            case 'tool_execution':
                addSystemMessage(`🔧 Using tool: ${event.data.tool}`)
                break

            case 'log':
                setCurrentActivity(event.data.message)
                break

            case 'answer':
                addAssistantMessage(event.data.answer)
                break

            case 'complete':
                setStatus(prev => ({ ...prev, processing: false }))
                setCurrentActivity('')
                break

            case 'error':
                addSystemMessage(`❌ Error: ${event.data.message}`)
                setStatus(prev => ({ ...prev, processing: false }))
                setCurrentActivity('')
                break
        }
    }

    const addSystemMessage = (content: string) => {
        const message: Message = {
            id: Date.now().toString(),
            role: 'system',
            content,
            timestamp: new Date(),
        }
        setMessages(prev => [...prev, message])
    }

    const addAssistantMessage = (content: string) => {
        const message: Message = {
            id: Date.now().toString(),
            role: 'assistant',
            content,
            timestamp: new Date(),
        }
        setMessages(prev => [...prev, message])
    }

    const handleSendMessage = (content: string, model?: string) => {
        if (!wsRef.current?.isConnected()) {
            alert('Not connected to server. Please wait...')
            return
        }

        // Add user message to chat
        const userMessage: Message = {
            id: Date.now().toString(),
            role: 'user',
            content,
            timestamp: new Date(),
        }
        setMessages(prev => [...prev, userMessage])

        // Send to backend with specified model or current model
        try {
            wsRef.current.sendMessage(content, model || status.model || undefined)
        } catch (error) {
            console.error('Error sending message:', error)
            addSystemMessage('Failed to send message. Please try again.')
        }
    }

    const handleModelChange = async (modelName: string) => {
        try {
            const response = await fetch('http://localhost:8000/api/select-model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_name: modelName }),
            })

            if (response.ok) {
                setStatus(prev => ({ ...prev, model: modelName }))
                addSystemMessage(`Switched to model: ${modelName}`)
            }
        } catch (error) {
            console.error('Error changing model:', error)
            addSystemMessage('Failed to change model')
        }
    }

    return (
        <div className="glass rounded-2xl p-6 shadow-2xl">
            {/* Status Bar */}
            <div className="flex items-center justify-between mb-6 pb-4 border-b border-dark-border">
                <StatusIndicator
                    connected={status.connected}
                    processing={status.processing}
                    activity={currentActivity}
                />
                <ModelSelector
                    currentModel={status.model}
                    onModelChange={handleModelChange}
                    disabled={status.processing}
                />
            </div>

            {/* Messages */}
            <MessageList messages={messages} />

            {/* Input */}
            <MessageInput
                onSend={handleSendMessage}
                disabled={!status.connected || status.processing}
                models={availableModels}
                currentModel={status.model}
            />
        </div>
    )
}
