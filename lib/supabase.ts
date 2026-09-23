import { createClient } from '@supabase/supabase-js'

// We provide empty strings as fallbacks so the build doesn't crash 
// even if Vercel hasn't loaded the envs yet during the linting phase.
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || ''
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || ''

export const supabase = createClient(supabaseUrl, supabaseAnonKey)