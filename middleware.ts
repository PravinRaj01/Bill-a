import { createServerClient } from '@supabase/ssr'
import { NextResponse, type NextRequest } from 'next/server'

export async function middleware(request: NextRequest) {
  let response = NextResponse.next({
    request: {
      headers: request.headers,
    },
  })

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return request.cookies.getAll()
        },
        setAll(cookiesToSet: any) {
          cookiesToSet.forEach(({ name, value }: any) => request.cookies.set(name, value))
          response = NextResponse.next({
            request,
          })
          cookiesToSet.forEach(({ name, value, options }: any) =>
            response.cookies.set(name, value, options)
          )
        },
      },
    }
  )

  const { data: { user } } = await supabase.auth.getUser()

  const path = request.nextUrl.pathname

  // --- NEW GUEST-FRIENDLY LOGIC ---
  
  // 1. Define routes that STRICTLY require a login
  const strictlyProtected = ['/dashboard/history', '/dashboard/account']
  const isStrictlyProtected = strictlyProtected.some(route => path.startsWith(route))

  // 2. If trying to access History/Account without a user, redirect to login
  if (!user && isStrictlyProtected) {
    return NextResponse.redirect(new URL('/', request.url))
  }

  // 3. If user is logged in and tries to go to the Landing/Login page, send to Dashboard
  if (user && path === '/') {
    return NextResponse.redirect(new URL('/dashboard', request.url))
  }

  // NOTE: If !user and path is /dashboard or /dashboard/new, it will now let them through!

  return response
}

export const config = {
  matcher: [
    // /dev/* is local diagnostic tooling (e.g. the WebLLM spike page) —
    // it has no business being auth-gated, and excluding it here means
    // testing it never depends on Supabase/Neon credentials being
    // configured at all.
    '/((?!_next/static|_next/image|favicon.ico|auth/callback|auth/login-google|api/scan|api/split|dev).*)',
  ],
}