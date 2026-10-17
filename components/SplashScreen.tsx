"use client"

export default function SplashScreen() {
  return (
    <div className="fixed inset-0 z-[100] flex flex-col items-center justify-center bg-background">
      <div className="relative">
        <div className="absolute -inset-10 rounded-full bg-primary/25 blur-3xl animate-pulse" />
        <div className="relative overflow-hidden rounded-3xl bg-white p-2 shadow-[var(--coral-glow)]">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/logo.png" alt="Bill.a" className="size-20 object-contain" />
        </div>
      </div>

      <div className="mt-8 text-center space-y-2">
        <h1 className="text-3xl font-black tracking-tighter text-foreground">
          Bill<span className="text-primary">.</span>a
        </h1>
        <p className="text-[10px] uppercase tracking-[0.5em] text-muted-foreground font-bold">The Splitter</p>
      </div>

      <div className="absolute bottom-16 w-32 h-[2px] bg-white/5 overflow-hidden rounded-full">
        <div className="w-full h-full bg-primary -translate-x-full animate-[loading_1.5s_infinite]" />
      </div>

      <style jsx>{`
        @keyframes loading {
          0% { transform: translateX(-100%); }
          50% { transform: translateX(0); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </div>
  )
}
