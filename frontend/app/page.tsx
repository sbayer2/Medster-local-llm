'use client'

import ChatInterface from '@/components/ChatInterface'

export default function Home() {
    return (
        <main className="min-h-screen p-4 md:p-8">
            <div className="max-w-6xl mx-auto">
                {/* Header */}
                <div className="mb-8 text-center">
                    <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-medical-400 to-medical-600 bg-clip-text text-transparent mb-2">
                        Medster Local LLM
                    </h1>
                    <p className="text-dark-text/70 text-sm md:text-base">
                        Medical Analysis Agent • Powered by Local LLMs • 100% Private
                    </p>
                </div>

                {/* Main Chat Interface */}
                <ChatInterface />
            </div>
        </main>
    )
}
