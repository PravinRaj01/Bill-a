"use client"

// Matches the V2 splash mock-up, scaled 1.7x from its 228px phone frame to a real screen.
export default function SplashScreen() {
  return (
    <div className="fixed inset-0 z-[100] flex flex-col items-center justify-center bg-background">
      <div className="relative grid place-items-center">
        {/* Coral halo behind the tile */}
        <div className="halo absolute -translate-y-[17px] size-[442px] rounded-full bg-[radial-gradient(circle,rgb(255_107_87/0.33),rgb(255_107_87/0)_65%)]" />
        {/* The app icon: the dark artwork on a dark tile */}
        <div className="relative size-[129px] overflow-hidden rounded-[26.3%] bg-background shadow-[0_17px_51px_-14px_rgb(255_107_87/0.6)]">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/logo-dark.png" alt="Bill.a" className="relative size-full object-contain p-2.5" />
        </div>
      </div>

      {/* System font (SF / Segoe UI / Roboto), as in the mock-up */}
      <div className="text-center font-[system-ui,-apple-system,Segoe_UI,Roboto,sans-serif]">
        <h1 className="mt-[44px] mb-[3px] text-[44px] leading-normal font-black tracking-[-0.05em] text-foreground">
          Bill<span className="text-primary">.</span>a
        </h1>
        <p className="m-0 text-[15px] uppercase tracking-[0.5em] text-muted-foreground font-bold">The Splitter</p>
      </div>

      <div className="absolute bottom-[82px] h-[3px] w-[164px] overflow-hidden rounded-[3px] bg-[#ffffff10]">
        <i className="bar block h-full bg-primary" />
      </div>

      <style jsx>{`
        .halo {
          animation: halo 2.4s ease-in-out infinite;
        }
        .bar {
          animation: loading 1.5s infinite;
        }
        @keyframes halo {
          50% { transform: scale(1.12); opacity: 0.7; }
        }
        @keyframes loading {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        @media (prefers-reduced-motion: reduce) {
          .halo, .bar { animation: none; }
        }
      `}</style>
    </div>
  )
}
