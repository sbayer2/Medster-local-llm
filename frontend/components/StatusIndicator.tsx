'use client'

interface StatusIndicatorProps {
    connected: boolean
    processing: boolean
    activity?: string
}

export default function StatusIndicator({ connected, processing, activity }: StatusIndicatorProps) {
    return (
        <div className="flex items-center gap-3">
            {/* Connection Status */}
            <div className="flex items-center gap-2">
                <div
                    className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500 animate-pulse-slow' : 'bg-red-500'
                        }`}
                />
                <span className="text-sm text-dark-text/70">
                    {connected ? 'Connected' : 'Disconnected'}
                </span>
            </div>

            {/* Processing Status */}
            {processing && (
                <div className="flex items-center gap-2 px-3 py-1 bg-medical-900/30 rounded-full">
                    <div className="flex gap-1">
                        <div className="w-1.5 h-1.5 bg-medical-400 rounded-full pulse-dot" />
                        <div className="w-1.5 h-1.5 bg-medical-400 rounded-full pulse-dot" />
                        <div className="w-1.5 h-1.5 bg-medical-400 rounded-full pulse-dot" />
                    </div>
                    <span className="text-xs text-medical-300">
                        {activity || 'Processing...'}
                    </span>
                </div>
            )}
        </div>
    )
}
