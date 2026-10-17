'use client'

import { useState, useEffect } from "react"
import Link from "next/link"
import SplashScreen from "@/components/SplashScreen"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { login, signup, loginWithGoogle } from "@/app/auth/actions"

export default function AuthPage() {
  const [showSplash, setShowSplash] = useState(true)
  const [isLogin, setIsLogin] = useState(true)
  const [error, setError] = useState<string | null>(null)
  // Controlled so React 19's automatic form reset (which runs after every form
  // action) doesn't wipe the email after a failed attempt. Password still clears.
  const [email, setEmail] = useState("")
  const [loading, setLoading] = useState(false)
  // The page a logged-out visitor was sent away from (an invite link, say): they land back on it after logging in.
  const [callbackUrl, setCallbackUrl] = useState("")

  // Initial App Load Splash Screen
  useEffect(() => {
    const timer = setTimeout(() => setShowSplash(false), 2500)
    return () => clearTimeout(timer)
  }, [])

  // Auth.js sends failures back as /?error=<code> (lib/auth.config.ts pages.error).
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const wanted = params.get("callbackUrl")
    if (wanted) queueMicrotask(() => setCallbackUrl(wanted))
    const code = params.get("error")
    if (!code) return
    queueMicrotask(() =>
      setError(
        code === "OAuthAccountNotLinked"
          ? "That email is already registered with a password. Log in with your email and password instead."
          : "Sign-in failed. Please try again.",
      ),
    )
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
    <div className="flex min-h-screen items-center justify-center p-4 animate-in fade-in duration-700">
      <div className="w-full max-w-md space-y-6 overflow-hidden rounded-3xl glass p-8 text-foreground ring-1 ring-white/5">
        <header className="space-y-2 text-center">
          <h1 className="type-title">{isLogin ? "Welcome back" : "Join Bill.a"}</h1>
          <p className="text-sm font-medium tracking-tight text-muted-foreground">
            {isLogin ? "Login to sync your split history" : "Create an account to save your groups"}
          </p>
        </header>

        {/* Google Authentication */}
        <form action={loginWithGoogle} className="w-full">
          <Button variant="outline" className="h-12 w-full rounded-xl border-white/10 bg-white/10 font-bold text-foreground transition-all hover:bg-white/10">
            <svg className="mr-2 h-4 w-4" viewBox="0 0 24 24">
              <path d="M12.48 10.92v3.28h7.84c-.24 1.84-.853 3.187-1.787 4.133-1.147 1.147-2.933 2.4-6.053 2.4-4.827 0-8.6-3.893-8.6-8.72s3.773-8.72 8.6-8.72c2.6 0 4.507 1.027 5.907 2.347l2.307-2.307C18.747 1.44 16.133 0 12.48 0 5.867 0 .307 5.387.307 12s5.56 12 12.173 12c3.573 0 6.267-1.173 8.373-3.36 2.16-2.16 2.84-5.213 2.84-7.667 0-.76-.053-1.467-.173-2.053H12.48z" fill="currentColor" />
            </svg>
            Continue with Google
          </Button>
        </form>

        <div className="flex items-center gap-4 text-xs tracking-[0.2em] uppercase">
          <span className="h-px flex-1 bg-white/10" />
          <span className="font-bold text-muted-foreground">Or use email</span>
          <span className="h-px flex-1 bg-white/10" />
        </div>

        {/* Credentials Form */}
        <form action={handleSubmit} className="grid gap-4">
          <input type="hidden" name="callbackUrl" value={callbackUrl} />
          {!isLogin && (
            <div className="grid gap-2">
              <Label htmlFor="name" className="ml-1 text-muted-foreground">
                Your name <span className="text-xs">(optional)</span>
              </Label>
              <Input id="name" name="name" type="text" autoComplete="name" maxLength={80} placeholder="What should we call you?" />
            </div>
          )}
          <div className="grid gap-2">
            <Label htmlFor="email" className="ml-1 text-muted-foreground">Email</Label>
            <Input
              id="email"
              name="email"
              type="email"
              placeholder="m@example.com"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
             
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="password" title="password" className="ml-1 text-muted-foreground">Password</Label>
            <Input id="password" name="password" type="password" required />
          </div>
          {error && <p role="alert" className="text-center text-sm font-medium text-coral">{error}</p>}
          <Button type="submit" className="h-12 w-full" disabled={loading}>
            {loading ? "Processing..." : (isLogin ? "Login" : "Create Account")}
          </Button>
        </form>

        <div className="text-center">
          <button
            type="button"
            onClick={() => setIsLogin(!isLogin)}
            className="min-h-11 text-sm font-medium text-muted-foreground underline underline-offset-4 transition-colors hover:text-foreground"
          >
            {isLogin ? "Need an account? Sign up" : "Already have an account? Login"}
          </button>
        </div>

        <div className="flex items-center gap-4 text-xs tracking-[0.2em] uppercase">
          <span className="h-px flex-1 bg-white/10" />
          <span className="font-bold text-muted-foreground/70">Incognito</span>
          <span className="h-px flex-1 bg-white/10" />
        </div>

        <Link href="/dashboard" className="block w-full">
          <Button type="button" variant="ghost" className="h-12 w-full type-eyebrow hover:bg-white/5 hover:text-foreground">
            Continue as Guest
          </Button>
        </Link>
      </div>
    </div>
  )
}
