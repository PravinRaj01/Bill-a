"use client"

export default function SplashScreen() {
  return (
    <div className="fixed inset-0 z-[100] flex flex-col items-center justify-center bg-[#09090b]">
      <div className="relative">
        <div className="absolute -inset-4 bg-white/5 rounded-full blur-2xl animate-pulse" />
        
        {/* YOUR LOGO HERE */}
        <div className="relative bg-white p-4 rounded-2xl shadow-2xl shadow-white/10 overflow-hidden">
          <img 
            src="/icon.png" 
            alt="Bill.a Logo" 
            className="w-12 h-12 object-contain"
          />
        </div>
      </div>
      
      <div className="mt-8 text-center space-y-2">
        <h1 className="text-3xl font-black tracking-tighter text-white">Bill.a</h1>
        <p className="text-[10px] uppercase tracking-[0.5em] text-slate-500 font-bold">The Splitter</p>
      </div>

      <div className="absolute bottom-16 w-32 h-[2px] bg-white/5 overflow-hidden rounded-full">
        <div className="w-full h-full bg-white/40 -translate-x-full animate-[loading_1.5s_infinite]" />
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