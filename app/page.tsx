'use client'

import { useState, useEffect } from "react"
import Link from "next/link"
import SplashScreen from "@/components/SplashScreen"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { login, signup } from "@/app/auth/actions"

export default function AuthPage() {
  const [showSplash, setShowSplash] = useState(true)
  const [isLogin, setIsLogin] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  // Initial App Load Splash Screen
  useEffect(() => {
    const timer = setTimeout(() => setShowSplash(false), 2500)
    return () => clearTimeout(timer)
  }, [])

  async function handleSubmit(formData: FormData) {
    setLoading(true)
    setError(null)
    const action = isLogin ? login : signup
    const result = await action(formData)
    if (result?.error) {
      setError(result.error)
      setLoading(false)
    }
  }

  if (showSplash) {
    return <SplashScreen />
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-black p-4 animate-in fade-in duration-700">
      <Card className="w-full max-w-md border-zinc-800 bg-zinc-950 text-white shadow-2xl rounded-3xl overflow-hidden ring-1 ring-white/5">
        <CardHeader className="text-center space-y-2 pt-8">
          <CardTitle className="text-3xl font-black tracking-tighter italic">
            {isLogin ? "Welcome back" : "Join Bill.a"}
          </CardTitle>
          <CardDescription className="text-zinc-500 font-medium tracking-tight">
            {isLogin 
              ? "Login to sync your split history" 
              : "Create an account to save your groups"}
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-6 pb-8">
          
          {/* Google Authentication */}
          <form action="/auth/login-google" method="POST" className="w-full">
            <Button variant="outline" className="w-full border-zinc-800 bg-zinc-900 text-white hover:bg-zinc-800 rounded-xl h-12 font-bold transition-all">
              <svg className="mr-2 h-4 w-4" viewBox="0 0 24 24">
                <path d="M12.48 10.92v3.28h7.84c-.24 1.84-.853 3.187-1.787 4.133-1.147 1.147-2.933 2.4-6.053 2.4-4.827 0-8.6-3.893-8.6-8.72s3.773-8.72 8.6-8.72c2.6 0 4.507 1.027 5.907 2.347l2.307-2.307C18.747 1.44 16.133 0 12.48 0 5.867 0 .307 5.387.307 12s5.56 12 12.173 12c3.573 0 6.267-1.173 8.373-3.36 2.16-2.16 2.84-5.213 2.84-7.667 0-.76-.053-1.467-.173-2.053H12.48z" fill="currentColor" />
              </svg>
              Continue with Google
            </Button>
          </form>

          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <span className="w-full border-t border-white/5" />
            </div>
            <div className="relative flex justify-center text-[10px] uppercase tracking-[0.2em]">
              <span className="bg-zinc-950 px-4 text-zinc-600 font-bold">Or use email</span>
            </div>
          </div>

          {/* Credentials Form */}
          <form action={handleSubmit} className="grid gap-4">
            <div className="grid gap-2">
              <Label htmlFor="email" className="text-zinc-400 ml-1">Email</Label>
              <Input 
                id="email" 
                name="email" 
                type="email" 
                placeholder="m@example.com" 
                required 
                className="bg-zinc-900 border-zinc-800 h-11 focus:ring-1 ring-white/20 transition-all" 
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="password" title="password" className="text-zinc-400 ml-1">Password</Label>
              <Input 
                id="password" 
                name="password" 
                type="password" 
                required 
                className="bg-zinc-900 border-zinc-800 h-11 focus:ring-1 ring-white/20 transition-all" 
              />
            </div>
            {error && <p className="text-sm text-red-500 text-center font-medium animate-bounce">{error}</p>}
            <Button type="submit" className="w-full bg-white text-black hover:bg-zinc-200 h-12 font-black uppercase tracking-tighter rounded-xl" disabled={loading}>
              {loading ? "Processing..." : (isLogin ? "Login" : "Create Account")}
            </Button>
          </form>

          <div className="text-center">
            <button 
              type="button"
              onClick={() => setIsLogin(!isLogin)} 
              className="text-xs text-zinc-500 hover:text-white transition-colors font-medium underline underline-offset-4"
            >
              {isLogin ? "Need an account? Sign up" : "Already have an account? Login"}
            </button>
          </div>

          <div className="relative pt-2">
             <div className="relative flex justify-center text-[10px] uppercase tracking-[0.2em]">
               <span className="bg-zinc-950 px-4 text-zinc-700 font-bold">Incognito</span>
             </div>
          </div>

          {/* --- FIXED GUEST LINK --- */}
          <Link href="/dashboard" className="w-full">
            <Button type="button" variant="ghost" className="w-full text-zinc-500 hover:text-white hover:bg-white/5 h-12 font-bold uppercase tracking-widest text-[10px]">
              Continue as Guest
            </Button>
          </Link>
        </CardContent>
      </Card>
    </div>
  )
}