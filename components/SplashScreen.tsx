"use client"

export default function SplashScreen() {
  return (
    <div className="fixed inset-0 z-[100] flex flex-col items-center justify-center bg-background">
      <div className="relative grid place-items-center">
        {/* Coral halo behind the tile */}
        <div className="absolute size-[28rem] rounded-full bg-[radial-gradient(circle,rgb(255_107_87/0.42),rgb(255_107_87/0)_65%)] animate-[halo_2.4s_ease-in-out_infinite] motion-reduce:animate-none" />
        {/* The app icon: the dark artwork on a dark tile */}
        <div className="relative size-[120px] overflow-hidden rounded-[26%] bg-background shadow-[0_10px_30px_-8px_rgb(255_107_87/0.6)]">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/logo-dark.png" alt="Bill.a" className="relative size-full object-contain p-2.5" />
        </div>
      </div>

      {/* System font (SF / Segoe UI / Roboto), as in the logo mock-ups */}
      <div className="mt-14 text-center font-[system-ui,-apple-system,Segoe_UI,Roboto,sans-serif]">
        <h1 className="text-[42px] leading-normal font-black tracking-[-0.05em] text-foreground">
          Bill<span className="text-primary">.</span>a
        </h1>
        <p className="mt-2 text-sm uppercase tracking-[0.5em] text-muted-foreground font-bold">The Splitter</p>
      </div>

      <div className="absolute bottom-20 w-40 h-[2px] bg-white/5 overflow-hidden rounded-full">
        <div className="w-full h-full bg-primary -translate-x-full animate-[loading_1.5s_infinite]" />
      </div>

      <style jsx>{`
        @keyframes halo {
          50% { transform: scale(1.12); opacity: 0.7; }
        }
        @keyframes loading {
          0% { transform: translateX(-100%); }
          50% { transform: translateX(0); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </div>
  )
}
