"use client"

export default function SplashScreen() {
  return (
    <div className="fixed inset-0 z-[100] flex flex-col items-center justify-center bg-background">
      <div className="relative grid place-items-center">
        {/* Coral halo behind the tile */}
        <div className="absolute size-72 rounded-full bg-[radial-gradient(circle,rgb(255_107_87/0.33),rgb(255_107_87/0)_65%)] animate-pulse" />
        {/* The app icon: dark tile with the dark artwork and a coral glow inside it */}
        <div className="relative size-32 overflow-hidden rounded-[28%] bg-background shadow-[0_10px_30px_-8px_rgb(255_107_87/0.6)]">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_60%,rgb(255_107_87/0.6)_30%,rgb(255_107_87/0)_78%)]" />
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/logo-dark.png" alt="Bill.a" className="relative size-full object-contain p-2.5" />
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
