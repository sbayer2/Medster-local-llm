'use client'

import { useState, KeyboardEvent } from 'react'
import { ModelInfo } from '@/lib/types'

interface MessageInputProps {
    onSend: (message: string, model?: string) => void
    disabled?: boolean
    models?: ModelInfo[]
    currentModel?: string | null
}

export default function MessageInput({ onSend, disabled, models = [], currentModel }: MessageInputProps) {
    const [input, setInput] = useState('')
    const [selectedModel, setSelectedModel] = useState<string | null>(null)
    const [isModelDropdownOpen, setIsModelDropdownOpen] = useState(false)

    // Use selected model or fall back to current global model
    const activeModel = selectedModel || currentModel || 'qwen3-vl:8b'
    const activeModelInfo = models.find(m => m.name === activeModel)

    // Filter to only show multimodal models
    const multimodalModels = models.filter(m => m.multimodal)

    const handleSend = () => {
        if (input.trim() && !disabled) {
            onSend(input.trim(), activeModel)
            setInput('')
            // Reset to global default after sending
            setSelectedModel(null)
        }
    }

    const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSend()
        }
    }

    return (
        <div className="space-y-2">
            {/* Model selector for this query */}
            {multimodalModels.length > 1 && (
                <div className="flex items-center gap-2 text-sm">
                    <span className="text-dark-text/60">Model for this query:</span>
                    <div className="relative">
                        <button
                            onClick={() => setIsModelDropdownOpen(!isModelDropdownOpen)}
                            disabled={disabled}
                            className="flex items-center gap-2 px-3 py-1.5 bg-dark-surface border border-dark-border rounded-lg hover:border-medical-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            <span className="text-sm font-medium text-dark-text">
                                {activeModelInfo?.name || 'Select Model'}
                            </span>
                            {activeModelInfo?.multimodal && (
                                <span className="text-xs bg-medical-500 text-white px-1.5 py-0.5 rounded">
                                    Vision
                                </span>
                            )}
                            <svg
                                className={`w-3 h-3 transition-transform ${isModelDropdownOpen ? 'rotate-180' : ''}`}
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                            >
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                            </svg>
                        </button>

                        {isModelDropdownOpen && (
                            <>
                                <div
                                    className="fixed inset-0 z-10"
                                    onClick={() => setIsModelDropdownOpen(false)}
                                />
                                <div className="absolute left-0 bottom-full mb-2 w-80 bg-dark-surface border border-dark-border rounded-lg shadow-xl z-20">
                                    {multimodalModels.map((model) => (
                                        <button
                                            key={model.name}
                                            onClick={() => {
                                                setSelectedModel(model.name)
                                                setIsModelDropdownOpen(false)
                                            }}
                                            className={`w-full text-left px-4 py-3 hover:bg-dark-bg transition-colors first:rounded-t-lg last:rounded-b-lg ${activeModel === model.name ? 'bg-medical-900/30' : ''
                                                }`}
                                        >
                                            <div className="flex items-start justify-between gap-2">
                                                <div>
                                                    <div className="font-semibold text-dark-text flex items-center gap-2">
                                                        {model.name}
                                                        {model.multimodal && (
                                                            <span className="text-xs bg-medical-500 text-white px-2 py-0.5 rounded">
                                                                Vision
                                                            </span>
                                                        )}
                                                    </div>
                                                    <div className="text-xs text-dark-text/60 mt-1">
                                                        {model.description}
                                                    </div>
                                                </div>
                                                {activeModel === model.name && (
                                                    <svg className="w-5 h-5 text-medical-500" fill="currentColor" viewBox="0 0 20 20">
                                                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                                    </svg>
                                                )}
                                            </div>
                                        </button>
                                    ))}
                                </div>
                            </>
                        )}
                    </div>
                </div>
            )}

            {/* Message input */}
            <div className="flex gap-3">
                <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={disabled ? 'Processing...' : 'Ask about patient data, labs, medications...'}
                    disabled={disabled}
                    rows={3}
                    className="flex-1 bg-dark-surface border border-dark-border rounded-xl px-4 py-3 text-dark-text placeholder-dark-text/40 focus:outline-none focus:ring-2 focus:ring-medical-500 focus:border-transparent resize-none disabled:opacity-50 disabled:cursor-not-allowed"
                />
                <button
                    onClick={handleSend}
                    disabled={disabled || !input.trim()}
                    className="px-6 py-3 bg-medical-gradient text-white rounded-xl font-semibold hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed self-end"
                >
                    Send
                </button>
            </div>
        </div>
    )
}
