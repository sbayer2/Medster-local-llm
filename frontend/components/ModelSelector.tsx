'use client'

import { useState, useEffect } from 'react'
import { ModelInfo } from '@/lib/types'

interface ModelSelectorProps {
    currentModel: string | null
    onModelChange: (model: string) => void
    disabled?: boolean
}

export default function ModelSelector({ currentModel, onModelChange, disabled }: ModelSelectorProps) {
    const [models, setModels] = useState<ModelInfo[]>([])
    const [isOpen, setIsOpen] = useState(false)

    useEffect(() => {
        fetchModels()
    }, [])

    const fetchModels = async () => {
        try {
            const response = await fetch('http://localhost:8000/api/models')
            const data = await response.json()
            setModels(data)
        } catch (error) {
            console.error('Error fetching models:', error)
        }
    }

    const currentModelInfo = models.find(m => m.name === currentModel)

    return (
        <div className="relative">
            <button
                onClick={() => setIsOpen(!isOpen)}
                disabled={disabled}
                className="flex items-center gap-2 px-4 py-2 bg-dark-surface border border-dark-border rounded-lg hover:border-medical-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
                <span className="text-sm font-medium text-dark-text">
                    {currentModelInfo?.name || 'Select Model'}
                </span>
                <svg
                    className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
            </button>

            {isOpen && (
                <>
                    <div
                        className="fixed inset-0 z-10"
                        onClick={() => setIsOpen(false)}
                    />
                    <div className="absolute right-0 mt-2 w-80 bg-dark-surface border border-dark-border rounded-lg shadow-xl z-20">
                        {models.map((model) => (
                            <button
                                key={model.name}
                                onClick={() => {
                                    onModelChange(model.name)
                                    setIsOpen(false)
                                }}
                                className={`w-full text-left px-4 py-3 hover:bg-dark-bg transition-colors first:rounded-t-lg last:rounded-b-lg ${currentModel === model.name ? 'bg-medical-900/30' : ''
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
                                    {currentModel === model.name && (
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
    )
}
